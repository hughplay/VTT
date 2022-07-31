import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import jsonlines
import ray
from ray.util.queue import Queue

sys.path.append(".")
from src.utils.raytool import ProgressBar  # noqa: E402

VTT_ROOT = Path("/data/vtt")
N_CPU_PER_THREAD = 8
FRAME_EXT = ".jpg"


# =============================================================================
# Helper functions
# =============================================================================


def get_frame_count(clip_path):
    from sh import ffprobe

    output = ffprobe(
        "-v",
        "error",
        "-show_entries",
        "stream=nb_frames",
        "-select_streams",
        "v:0",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(clip_path),
    )
    return int(output)


def are_frames_full(clip_path, save_root):
    frame_count = get_frame_count(clip_path)
    return len(list(Path(save_root).glob(f"*{FRAME_EXT}"))) == frame_count


def get_resized_width_height(resolution, width, height):

    if width < height:
        new_width = int(resolution)
        new_height = int(height / width * new_width)
    else:
        new_height = int(resolution)
        new_width = int(width / height * new_height)

    return (new_width, new_height)


def get_existed_frames(video_path):
    exist_frames = list(video_path.glob(f"*{FRAME_EXT}"))
    return len(exist_frames)


def save_frames_opencv(clip_path, save_root, resolution, logger=None):
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip_path))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = get_frame_count(clip_path)

    if list(save_root.glob(FRAME_EXT)) != frame_count:
        for i in range(frame_count):
            frame_path = save_root / f"{i:04d}{FRAME_EXT}"
            if not frame_path.exists():
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(
                            frame,
                            dsize=get_resized_width_height(
                                resolution, width, height
                            ),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        frame_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(frame_path), frame)
                    else:
                        if logger is not None:
                            logger.warning(f"empty frame: {frame_path}")
                except Exception:
                    if logger is not None:
                        logger.exception(f"{frame_path} failed:")


def save_frames_ffmpeg(clip_path, save_root, resolution):
    from sh import ffmpeg

    save_root = Path(save_root)

    # remove all images from save_root
    if save_root.exists():
        save_root.rmdir(recursive=True)
    save_root.mkdir(parents=True, exist_ok=True)

    output = ffmpeg(
        "-y",
        "-i",
        f"{clip_path}",
        "-vf",
        rf"scale=if(gte(iw\,ih)\,-1\,{resolution}):if(gte(iw\,ih)\,{resolution}\,-1)",
        "-hide_banner",
        "-loglevel",
        "error",
        f"{save_root / '%04d'}{FRAME_EXT}",
    )
    return output


# =============================================================================
# ray remote function for extracting frames
# =============================================================================


@ray.remote(num_cpus=N_CPU_PER_THREAD)
def generate_frames(jobs, completed_queue, args, actor=None):
    from loguru import logger

    logger.configure(
        handlers=[
            # {"sink": sys.stdout, "colorize": True},
            {"sink": args.log, "enqueue": True},
        ]
    )

    while not jobs.empty():
        sample = jobs.get()

        sample_success = True
        steps = sample["annotation"]

        clip_path_list = []
        missing_clip = None
        for i in range(len(steps)):
            step_path = (
                Path(args.source) / f"{sample['id']}_{len(steps)}_{i}.mp4"
            )
            clip_path_list.append(step_path)
            if not step_path.exists():
                missing_clip = step_path
                break

        if missing_clip is None:
            try:

                logger.info(f"start to extract frames for {sample['id']}")

                sample["frames"] = {}
                # iterate through the clips and extract frames
                for clip_path in clip_path_list:
                    out_dir = Path(args.output) / f"{clip_path.stem}"

                    if not are_frames_full(clip_path, out_dir):
                        try:
                            logger.info(
                                f"Using ffmpeg to extract frames from {clip_path}"
                            )
                            save_frames_ffmpeg(
                                clip_path, out_dir, args.resolution
                            )

                        except KeyboardInterrupt:
                            raise

                        except Exception as e:
                            logger.exception(f"ffmpeg failed: \n{e}")

                    if not are_frames_full(clip_path, out_dir):
                        try:
                            logger.info(
                                f"Using OpenCV to extract frames from {clip_path}"
                            )
                            result = save_frames_opencv(
                                clip_path, out_dir, args.resolution, logger
                            )
                        except KeyboardInterrupt:
                            raise

                        except Exception as e:
                            logger.exception(f"OpenCV failed: \n{e}")

                    result = {
                        "frames": get_frame_count(clip_path),
                        "imgs": len(list(out_dir.glob(f"*{FRAME_EXT}"))),
                    }
                    sample["frames"][clip_path.stem] = result
                    if result["frames"] != result["imgs"]:
                        logger.warning(
                            f"{clip_path.stem} has {result['frames']} frames "
                            f"but {len(result['imgs'])} images"
                        )

            except KeyboardInterrupt:
                logger.warning("Keyboard Interrupt.")
                sample["status"] = "interrupted"
                sys.exit(0)

            except Exception as e:
                logger.exception(f"failed: {sample['id']}")
                sample["status"] = "error"
                sample["error"] = str(e)
                pass

            if sample_success:
                sample["status"] = "success"
            else:
                sample["status"] = "failed"

        else:
            logger.warning(f"{missing_clip} doesn't exist.")
            sample["status"] = "missing"

        completed_queue.put(sample)

        logger.info(
            f"{sample['id']}: {sample['status']}. samples leave: {len(jobs)}"
        )

        if actor:
            actor.update.remote(1)


# =============================================================================
# main function
# =============================================================================


def main(args):

    from loguru import logger

    logger.configure(
        handlers=[
            {"sink": sys.stdout, "colorize": True},
            {"sink": args.log, "enqueue": True},
        ]
    )

    Path(args.output).mkdir(parents=True, exist_ok=True)

    logger.info(f"Number of Threads: {args.threads}.")
    ray.init(num_cpus=args.threads * N_CPU_PER_THREAD)

    with jsonlines.open(args.list) as reader:
        samples = list(reader)
    n_sample = len(samples)
    logger.info(f"Total samples: {n_sample}")

    jobs_queue = Queue()
    for sample in samples:
        jobs_queue.put(sample)
    # jobs_list = split_list(samples, args.threads, by_chunk=False)
    complete_queue = Queue()

    pb = ProgressBar(n_sample)
    actor = pb.actor

    job_list = []

    for i in range(args.threads):
        job_list.append(
            # generate_states.remote(jobs_list[i], args, actor)
            generate_frames.remote(jobs_queue, complete_queue, args, actor)
        )

    try:
        pb.print_until_done()
        ray.get(job_list)
        ray.get(actor.get_counter.remote())

    except KeyboardInterrupt:
        logger.warning("Keyboard Interrupt.")

    except Exception as e:
        logger.exception(f"failed: {e}")

    finally:

        job_results = []
        while not complete_queue.empty():
            job_results.append(complete_queue.get())

        report_path = Path(args.report)
        logger.info(f"writing report to {report_path}")
        with jsonlines.open(report_path, "w") as writer:
            writer.write_all(job_results)

        status = defaultdict(int)
        for job in job_results:
            status[job["status"]] += 1
        for s, n in sorted(status.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{s}: {n}")

        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--list",
        default=VTT_ROOT / "meta" / "vtt_pre.jsonl",
        help="vtt list",
    )
    parser.add_argument(
        "-i",
        "--source",
        default=VTT_ROOT / "clips",
        help="directory of videos",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=VTT_ROOT / "frames",
        help="output directory",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        default=256,
        help="output resolution",
    )
    parser.add_argument(
        "--log",
        default=VTT_ROOT / "log" / "extract_frames.log",
        help="log path",
    )
    parser.add_argument(
        "--report",
        default=VTT_ROOT / "meta" / "report_extract_frames.jsonl",
        help="out vtt list with status",
    )
    parser.add_argument("--threads", default=8, type=int, help="total threads")
    args = parser.parse_args()

    main(args)
