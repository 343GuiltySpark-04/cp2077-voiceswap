import asyncio
import ctypes
import logging
import os
from functools import partial
from inspect import currentframe, getframeinfo
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING, Optional


from librosa.util.exceptions import ParameterError
from rich.progress import Progress, TaskID
from torch.multiprocessing import JoinableQueue, Process, Queue, Value
from tqdm import tqdm

import config
import lib.ffmpeg as ffmpeg
from util import Parallel, find_files
from util.rich_console import console, create_progress


if TYPE_CHECKING:
    from audio_separator.separator import Separator


old_tqdm_init = tqdm.__init__


def new_tqdm_init(*args, **kwargs):
    """Disable tqdm progress bar for audio_separator."""
    frame = currentframe()
    if frame is not None:
        frame = frame.f_back

    disable = False

    while frame is not None:
        try:
            frame_info = getframeinfo(frame)
        except (OSError, RuntimeError):
            frame = frame.f_back
            continue

        if "audio_separator" in Path(frame_info.filename).parts:
            disable = True
            break

        frame = frame.f_back

    if disable:
        kwargs.setdefault("disable", True)

    old_tqdm_init(*args, **kwargs)



tqdm.__init__ = new_tqdm_init


def _custom_final_process(
    output_path: str, filename: str, self, _stem_path, source, stem_name
):
    os.makedirs(output_path, exist_ok=True)
    self.write_audio(
        os.path.join(output_path, f"{filename}_{stem_name.lower()}.wav"),
        source,
    )

    return {stem_name: source}


class UVRProcess(Process):
    """Process for running UVR."""

    def __init__(self, queue: Queue = None, progress_counter: Value = None, **kwargs):
        Process.__init__(self, **kwargs)

        self._run = Value(ctypes.c_bool, False)
        self._queue = queue or JoinableQueue()
        self._progress_counter = progress_counter or Value(ctypes.c_int, 0)
        self._last_model: Optional[str] = None
        self._separator: Optional["Separator"] = None

    def _separate(self, input_path: str, output_path: str, file: str, attempt=0):
        dirname = os.path.dirname(file)
        filename = os.path.basename(file)

        self._separator.model_instance.final_process = partial(
            _custom_final_process,
            os.path.join(output_path, dirname),
            filename,
            self._separator.model_instance,
        )

        try:
            return self._separator.separate(os.path.join(input_path, file))
        except ParameterError:
            if attempt < 3:
                return self._separate(input_path, output_path, file, attempt + 1)
            raise

    def terminate(self):
        if self._run:
            self._run = False
        else:
            Process.terminate(self)

    def run(self):
        from audio_separator.separator import Separator

        self._separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=config.UVR_MODEL_CACHE,
            mdx_params={
                "hop_length": 1024,
                "segment_size": 512,
                "overlap": 0.5,
                "batch_size": 1,
                "enable_denoise": True,
            },
            vr_params={
                "batch_size": 4,
                "window_size": 320,
                "aggression": 5,
                "enable_tta": False,
                "enable_post_process": False,
                "post_process_threshold": 0.2,
                "high_end_process": False,
            },
        )

        while self._run:
            try:
                input_path, output_path, file, wanted_model = self._queue.get(
                    timeout=0.1
                )
            except (Empty, TimeoutError):
                continue

            try:
                if wanted_model != self._last_model:
                    self._separator.load_model(wanted_model)
                    self._last_model = wanted_model

                self._separate(input_path, output_path, file)

                with self._progress_counter.get_lock():
                    self._progress_counter.value += 1
            except Exception:
                self._queue.put((input_path, output_path, file, wanted_model))
                raise
            finally:
                self._queue.task_done()


class UVRProcessManager:
    """Manage multiple UVR processes."""

    def __init__(self, jobs=1):
        self._queue = JoinableQueue()
        self._progress_counter = Value(ctypes.c_int, 0)
        self._wanted_model: Optional[str] = None

        self._workers = set(
            UVRProcess(self._queue, self._progress_counter) for _ in range(jobs)
        )

        self._progress_display: Optional[Progress] = None
        self._progress_task_id: Optional[TaskID] = None

        for worker in self._workers:
            worker.start()

    def submit(self, input_path: str, output_path: str, file: str):
        """Submit work to workers."""
        if self._wanted_model is None:
            raise ValueError("No model has been set yet.")

        self._queue.put((input_path, output_path, file, self._wanted_model))

    def set_model(self, model: str):
        """Change to loaded model."""
        self._wanted_model = model

    def configure_progress(
        self,
        progress: Progress,
        task_id: TaskID,
        *,
        total: Optional[int] = None,
        visible: Optional[bool] = None,
    ):
        """Attach a Rich progress task to track worker progress."""
        self._progress_display = progress
        self._progress_task_id = task_id

        with self._progress_counter.get_lock():
            self._progress_counter.value = 0

        update_kwargs = {"completed": 0}
        if total is not None:
            update_kwargs["total"] = total
        if visible is not None:
            update_kwargs["visible"] = visible

        self._progress_display.update(task_id, **update_kwargs)

    def wait(self):
        """Wait for all workers to finish."""
        self._queue.join()


    def _get_progress_task(self):
        """Retrieve the current rich progress task with backward compatibility."""
        if self._progress_display is None or self._progress_task_id is None:
            raise RuntimeError("Progress has not been configured.")

        progress = self._progress_display
        task_id = self._progress_task_id

        get_task = getattr(progress, "get_task", None)
        if callable(get_task):
            return get_task(task_id)

        tasks = getattr(progress, "tasks", None)
        if tasks is not None:
            for task in tasks:
                if getattr(task, "id", None) == task_id:
                    return task

        raise AttributeError("Progress object does not support retrieving tasks by id.")

    async def watch(self):
        """Wait and update the progress asynchronously."""
        task = self._get_progress_task()
        if task.total is not None and task.total == 0:
            return

        while True:
            increment = 0

            with self._progress_counter.get_lock():
                if self._progress_counter.value:
                    increment = self._progress_counter.value
                    self._progress_counter.value = 0

            if increment:
                self._progress_display.advance(self._progress_task_id, increment)

            for worker in [*self._workers]:
                if worker.exitcode is not None:
                    console.log("[yellow]WARNING[/]: A worker died, respawning...")
                    self._workers.remove(worker)
                    new_worker = UVRProcess(self._queue, self._progress_counter)
                    new_worker.start()
                    self._workers.add(new_worker)

            task = self._get_progress_task()
            if task.total is not None and task.completed >= task.total:
                break

            await asyncio.sleep(0.05)

    def terminate(self):
        """Terminate all workers."""
        for worker in self._workers:
            worker.terminate()

    def join(self):
        """Join all workers."""
        for worker in self._workers:
            worker.join()


async def isolate_vocals(
    input_path: str,
    cache_path=config.CACHE_PATH,
    overwrite: bool = True,
    n_workers=1,
):
    """Splits audio files to vocals and the rest. The audio has to be correct wav."""
    formatted_path = os.path.join(cache_path, config.UVR_FORMAT_CACHE)
    split_path = os.path.join(cache_path, config.UVR_FIRST_CACHE)
    reverb_path = os.path.join(cache_path, config.UVR_SECOND_CACHE)

    files = set(find_files(input_path))

    if not overwrite:
        skipped = 0

        for file in list(files):
            output_file = file.replace(".ogg", ".wav") + config.UVR_SECOND_SUFFIX
            if os.path.exists(os.path.join(reverb_path, output_file)):
                skipped += 1
                files.remove(file)

        console.log(f"Skipping {skipped} already done files.")

    if len(files) == 0:
        console.log("No files to process.")
        return

    with create_progress() as progress:
        ffmpegs = Parallel(
            "[Phase 1/3] Converting files",
            unit="files",
            progress=progress,
        )
        split_task_id = progress.add_task(
            "[Phase 2/3] Separating audio",
            total=0,
            unit="files",
            visible=False,
        )
        reverb_task_id = progress.add_task(
            "[Phase 3/3] Removing reverb",
            total=0,
            unit="files",
            visible=False,
        )

        uvr_workers = UVRProcessManager(n_workers)
        uvr_workers.set_model(config.UVR_FIRST_MODEL)
        uvr_workers.configure_progress(
            progress,
            split_task_id,
            total=0,
            visible=False,
        )

        async def convert_and_process(file: str):
            dirname = os.path.dirname(file)
            os.makedirs(os.path.join(formatted_path, dirname), exist_ok=True)

            converted_file = file.replace(".ogg", ".wav")
            converted_path = os.path.join(formatted_path, converted_file)
            if overwrite or not os.path.exists(converted_path):
                await ffmpeg.to_wav(os.path.join(input_path, file), converted_path)

            uvr_workers.submit(formatted_path, split_path, converted_file)

        split_files = []

        for file in files:
            output_file = file.replace(".ogg", ".wav") + config.UVR_FIRST_SUFFIX

            if not overwrite:
                output = os.path.join(split_path, output_file)
                if os.path.exists(output):
                    continue

            split_files.append(output_file)
            ffmpegs.run(convert_and_process, file)

        cached = len(files) - len(split_files)

        if not overwrite and cached > 0:
            console.log(f"Won't split {cached} already split files.")

        uvr_workers.configure_progress(
            progress,
            split_task_id,
            total=len(split_files),
            visible=len(split_files) > 0,
        )

        await asyncio.gather(ffmpegs.wait(), uvr_workers.watch())
        progress.update(split_task_id, visible=False)

        console.log("Waiting for workers...")
        uvr_workers.wait()

        reverb_total = len(files)
        uvr_workers.set_model(config.UVR_SECOND_MODEL)
        uvr_workers.configure_progress(
            progress,
            reverb_task_id,
            total=reverb_total,
            visible=reverb_total > 0,
        )

        for file in files:
            input_file = file.replace(".ogg", ".wav") + config.UVR_FIRST_SUFFIX
            uvr_workers.submit(split_path, reverb_path, input_file)

        await uvr_workers.watch()
        progress.update(reverb_task_id, visible=False)

        console.log("Waiting for workers...")
        uvr_workers.wait()

        uvr_workers.terminate()
        uvr_workers.join()
