import argparse
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path

import cv2
import jsonlines
import ray
from ray.util.queue import Queue
from sh import ffmpeg

sys.path.append(".")
from src.utils.raytool import ProgressBar  # noqa: E402

VTT_ROOT = Path("/data/reason/vtt")
N_CPU_PER_THREAD = 8


# =============================================================================
# Helper functions
# =============================================================================


def is_valid_file(path):
    path = Path(path)
    is_valid = path.exists() and path.stat().st_size > 0
    return is_valid


def save_clip_ffmpeg(video_path, save_path, start, end):
    output = ffmpeg(
        "-i",
        f"{video_path}",
        "-ss",
        f"{start}",
        "-to",
        f"{end}",
        "-hide_banner",
        "-loglevel",
        "error",
        f"{save_path}",
    )
    return output


# =============================================================================
# ray remote function for extracting clips
# =============================================================================


@ray.remote(num_cpus=N_CPU_PER_THREAD)
def generate_clips(jobs, completed_queue, args, actor=None):
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

        video_path = Path(args.source) / f"{sample['youtube_id']}.mp4"

        if video_path.exists():

            try:

                logger.info(
                    f"start to extract clips for {sample['id']}: {video_path}"
                )

                # iterate through the steps and cut the clips
                for i, step in enumerate(steps):
                    start, end = step["segment"]

                    out_path = (
                        Path(args.output)
                        / f"{sample['id']}_{len(steps)}_{i}.mp4"
                    )
                    method = "skip"

                    # skip if the clip already exists
                    if not is_valid_file(out_path):
                        method = "ffmpeg"
                        try:
                            save_clip_ffmpeg(video_path, out_path, start, end)
                        except Exception as e:
                            logger.exception(f"failed: {e}")

                    # check again
                    if not is_valid_file(out_path):
                        logger.warning(f"generating {out_path} failed.")
                        sample_success = False
                    else:
                        logger.info(f"method ({method}): {out_path.name}")

            except KeyboardInterrupt:
                logger.warning("Keyboard Interrupt.")
                sample["status"] = "interrupted"
                sys.exit(0)

            except Exception as e:
                logger.exception(f"failed: {video_path}")
                sample["status"] = "error"
                sample["error"] = str(e)
                pass

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
            generate_clips.remote(jobs_queue, complete_queue, args, actor)
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
        default=VTT_ROOT / "clips",
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
        default=VTT_ROOT / "log" / "extract_clips.log",
        help="log path",
    )
    parser.add_argument(
        "--report",
        default=VTT_ROOT / "meta" / "report_extract_clips.jsonl",
        help="out vtt list with status",
    )
    parser.add_argument("--threads", default=8, type=int, help="total threads")
    args = parser.parse_args()

    main(args)
