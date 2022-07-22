import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import jsonlines
import ray
from ray.util.queue import Queue
from sh import ffmpeg, ffprobe

from src.utils.raytool import ProgressBar

VTT_ROOT = Path("/data/reason/vtt")
N_CPU_PER_THREAD = 8


# =============================================================================
# Helper functions
# =============================================================================


def is_valid_file(path):
    path = Path(path)
    is_valid = path.exists() and path.stat().st_size > 0
    return is_valid


def get_resized_width_height(resolution, width, height):

    if width < height:
        new_width = int(resolution)
        new_height = int(height / width * new_width)
    else:
        new_height = int(resolution)
        new_width = int(width / height * new_height)

    return (new_width, new_height)


# only used when opencv failed
def save_frame_ffmpeg(video_path, save_path, time, resolution):
    output = ffmpeg(
        "-y",
        "-i",
        f"{video_path}",
        "-ss",
        f"{time}",
        "-vf",
        rf"scale=if(gte(iw\,ih)\,-1\,{resolution}):if(gte(iw\,ih)\,{resolution}\,-1)",
        "-vframes",
        "1",
        "-hide_banner",
        "-loglevel",
        "error",
        f"{save_path}",
    )
    return output


def get_duration(video_path):
    duration = ffprobe(
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    )
    return float(duration)


def save_frame_opencv(cap, t_frame, frame_path, resolution, logger=None):

    fps = cap.get(cv2.CAP_PROP_FPS)
    max_index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    target_index = min(int(fps * t_frame), max_index)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # if extracting target frame failed, then consider 30 frames before and
    # after the target frame (1 seconds) as a replacement.
    candidate_indices = [target_index]
    relax_seconds = 1
    for i in range(int(fps) * relax_seconds):
        candidate_indices.append(target_index - i)
        candidate_indices.append(target_index + i)

    for frame_index in candidate_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if ret:

            frame = cv2.resize(
                frame,
                dsize=get_resized_width_height(resolution, width, height),
                interpolation=cv2.INTER_CUBIC,
            )

            frame_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_path), frame)

            return True

    if logger is not None:
        print_func = logger.info

    else:
        print_func = print

    print_func(
        f"fps: {fps}, "
        f"failed, frame time: {t_frame}, "
        f"target index: {target_index}, "
        f"max index: {max_index}, "
        f"tried index: {[candidate_indices]}"
    )
    return False


# =============================================================================
# ray remote function for extracting states
# =============================================================================


@ray.remote(num_cpus=N_CPU_PER_THREAD)
def generate_states(jobs, completed_queue, args, actor=None):
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
        t_states = [steps[0]["segment"][0]] + [
            step["segment"][1] for step in steps
        ]

        video_path = Path(args.source) / f"{sample['youtube_id']}.mp4"

        if video_path.exists():

            try:

                sample["duration"] = get_duration(video_path)
                logger.info(
                    f"start to extract states for {sample['id']} "
                    f"({sample['duration']}): {video_path}"
                )

                # get the video capture object
                cap = cv2.VideoCapture(str(video_path))

                # iterate through the steps and extract the frames
                for i, t_state in enumerate(t_states):

                    out_path = (
                        Path(args.output)
                        / f"{sample['id']}_{len(t_states)}_{i}.jpg"
                    )
                    method = "skip"

                    # skip if the frame already exists
                    if not is_valid_file(out_path):
                        method = "opencv"
                        try:
                            save_frame_opencv(
                                cap,
                                t_state,
                                out_path,
                                args.resolution,
                                logger=logger,
                            )

                        except KeyboardInterrupt:
                            raise

                        except Exception:
                            logger.exception(
                                f"OpenCV failed for: {out_path.stem}"
                            )
                            pass

                    # if OpenCV failed, then use ffmpeg to extract the frame
                    if not is_valid_file(out_path):
                        method = "ffmpeg"
                        try:
                            save_frame_ffmpeg(
                                video_path,
                                out_path,
                                t_state,
                                args.resolution,
                            )

                        except KeyboardInterrupt:
                            raise

                        except Exception:
                            logger.exception(
                                f"FFmpeg failed for: {out_path.stem}"
                            )
                            pass

                    # check again
                    if not is_valid_file(out_path):
                        logger.warning(f"generating {out_path} failed.")
                        sample_success = False
                    else:
                        logger.info(f"method ({method}): {out_path.name}")

            except KeyboardInterrupt:
                logger.warning("Keyboard Interrupt.")
                sample["status"] = "interrupted"
                raise

            except Exception as e:
                logger.exception(f"failed: {video_path}")
                sample["status"] = "error"
                sample["error"] = str(e)
                pass

            finally:
                # release the video capture
                cap.release()

            if sample_success:
                sample["status"] = "success"
            else:
                sample["status"] = "failed"

        else:
            logger.warning(f"{video_path} doesn't exist.")
            sample["status"] = "missing"

        completed_queue.put(sample)

        logger.info(
            f"{sample['id']}: {sample['status']}. samples leave: {len(jobs)}"
        )

        if actor:
            actor.update.remote(1)


# =============================================================================
# main funciton
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
            generate_states.remote(jobs_queue, complete_queue, args, actor)
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
        default=VTT_ROOT / "meta" / "vtt.jsonl",
        help="vtt list",
    )
    parser.add_argument(
        "-i",
        "--source",
        default=VTT_ROOT / "videos",
        help="directory of videos",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=VTT_ROOT / "states",
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
        default=VTT_ROOT / "log" / "extract_states.log",
        help="log path",
    )
    parser.add_argument(
        "--report",
        default=VTT_ROOT / "meta" / "report_extract_states.jsonl",
        help="out vtt list with status",
    )
    parser.add_argument("--threads", default=8, type=int, help="total threads")
    args = parser.parse_args()

    main(args)
