from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import config
import lib.ffmpeg as ffmpeg

from util import Parallel, SubprocessException, find_files, spawn
from util.rich_console import console, create_progress

if TYPE_CHECKING:
    from rich.progress import Progress


async def _poetry_get_venv(path: str) -> str:
    """Return the path to the Poetry-managed virtual environment."""

    process = await spawn(
        "Poetry",
        "poetry",
        "env",
        "list",
        "--full-path",
        cwd=path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _stderr = await process.communicate()
    lines = stdout.splitlines()
    if not lines:
        raise RuntimeError("Poetry did not return any virtual environments.")
    return lines[0].decode()


async def _get_rvc_executable() -> str:
    """Resolve the Python executable for the RVC environment."""
    rvc_path = os.getenv("RVC_PATH")
    if not rvc_path:
        raise RuntimeError("RVC_PATH environment variable is not set.")

    venv = os.getenv("RVC_VENV")
    if not venv:
        venv = await _poetry_get_venv(rvc_path)

    return os.path.join(rvc_path, venv, "python")


UVR_DONE_TOKEN = "##done##"


class UVR:
    """Handle communication with the UVR isolation subprocess."""

    def __init__(self, model: str, input_path: str, output_path: str, batchsize: int):
        self.model = model
        self.input_path = input_path
        self.output_path = output_path
        self.batchsize = batchsize

        self.process: Optional[asyncio.subprocess.Process] = None
        self._progress: Optional["Progress"] = None
        self._task_id: Optional[int] = None

    async def start(self, total: int, title: str = "Isolating vocals") -> None:
        """Start the UVR subprocess and initialize progress tracking."""
        cwd = Path.cwd()
        progress = create_progress()
        progress.start()
        self._progress = progress
        self._task_id = progress.add_task(title, total=total, unit="file")

        script_path = cwd / "libs" / "rvc_uvr.py"
        input_dir = cwd / self.input_path
        output_dir = cwd / self.output_path

        try:
            self.process = await spawn(
                "RVC's venv python",
                await _get_rvc_executable(),
                str(script_path),
                self.model,
                str(input_dir),
                str(output_dir),
                str(self.batchsize),
                cwd=os.getenv("RVC_PATH"),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
        except Exception:
            self._stop_progress()
            raise

    async def run(self) -> None:
        """Monitor the UVR subprocess for progress updates."""
        process = self._ensure_process()

        try:
            while True:
                if process.returncode is not None:
                    break

                try:
                    line = await asyncio.wait_for(process.stdout.readline(), 5)
                except asyncio.TimeoutError:
                    stdin = process.stdin
                    if stdin and not stdin.is_closing():
                        stdin.write(b"\n")
                        await stdin.drain()
                    continue

                if not line:
                    continue

                message = line.decode().strip()
                if not message:
                    continue

                if message == UVR_DONE_TOKEN:
                    self._advance_progress()
                else:
                    console.log(message)
        finally:
            if process.returncode is None:
                await process.wait()
            self._stop_progress()

        if process.returncode != 0:
            raise SubprocessException(
                f"UVR process exited with code: {process.returncode}"
            )

    async def submit(self, file: str) -> None:
        """Submit a file to the UVR subprocess."""
        process = self._ensure_process()
        stdin = process.stdin
        if stdin is None:
            raise RuntimeError("UVR process stdin is not available.")
        stdin.write(f"{file}\n".encode())
        await stdin.drain()

    def _ensure_process(self) -> asyncio.subprocess.Process:
        if self.process is None:
            raise RuntimeError("UVR process has not been started.")
        return self.process

    def _advance_progress(self, step: int = 1) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=step)

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None


async def isolate_vocals(
    input_path: str,
    output_path: str,
    overwrite: bool = True,
    batchsize: int = 1,
    cache_path: str = config.TMP_PATH,
) -> None:
    """Split audio files into vocal and instrumental stems using UVR."""
    files: list[str]

    if overwrite:
        files = list(find_files(input_path))
    else:
        files = []
        skipped = 0
        for file in find_files(input_path):
            filename = os.path.basename(file)
            output_filename = _get_output_filename(filename, config.UVR_FIRST_MODEL)
            if os.path.exists(os.path.join(output_path, output_filename)):
                skipped += 1
            else:
                files.append(file)
        if skipped:
            console.log(f"[yellow]Skipping {skipped} already processed files.[/]")

    if not files:
        console.log("No files to process.")
        return

    formatted_path = os.path.join(cache_path, "formatted")
    split_path = os.path.join(cache_path, "split")

    uvr_split = UVR(config.UVR_FIRST_MODEL, formatted_path, split_path, batchsize)
    ffmpegs = Parallel("[Phase 1/3] Converting files", leave=True)

    async def convert_and_process(file: str) -> None:
        dirname = os.path.dirname(file)
        os.makedirs(os.path.join(formatted_path, dirname), exist_ok=True)

        output_file = file.replace(".ogg", ".wav")
        output_wav = os.path.join(formatted_path, output_file)
        if overwrite or not os.path.exists(output_wav):
            await ffmpeg.to_wav(os.path.join(input_path, file), output_wav)

        await uvr_split.submit(output_file)

    cached = 0
    for file in files:
        if not overwrite:
            filename = os.path.basename(file)
            dirname = os.path.dirname(file)
            output_filename = filename.replace(".ogg", ".wav")
            output_filename = _get_output_filename(
                output_filename, config.UVR_FIRST_MODEL
            )
            if os.path.exists(os.path.join(split_path, dirname, output_filename)):
                cached += 1
                continue

        ffmpegs.run(convert_and_process, file)

    if cached:
        console.log(f"[yellow]Skipping {cached} already split files.[/]")

    await uvr_split.start(ffmpegs.count_jobs(), "[Phase 2/3] Splitting vocals")
    await asyncio.gather(ffmpegs.wait(), uvr_split.run())

    uvr_dereverb = UVR(
        config.UVR_SECOND_MODEL,
        formatted_path,
        split_path,
        batchsize / 4,
    )
    await uvr_dereverb.start(len(files), "[Phase 3/3] Removing reverb")
    for file in files:
        await uvr_dereverb.submit(file)
    await uvr_dereverb.run()

    console.log("Cleaning up...")
    shutil.rmtree(config.TMP_PATH, ignore_errors=True)


def _get_output_filename(filename: str, model: str, instrument: bool = False) -> str:
    """Return the expected output filename for a UVR-processed file."""

    if model == "onnx_dereverb_By_FoxJoy":
        suffix = "others" if instrument else "vocal"
        return f"{filename}_{suffix}.wav"

    agg = 15  # set by the RVC script
    prefix = "instrument" if instrument else "vocal"
    return f"{prefix}_{filename}_{agg}.wav"


async def batch_rvc(
    input_path: str,
    opt_path: str,
    overwrite: bool,
    **kwargs: Any,
) -> None:
    """Run RVC over the given folder."""

    cwd = Path.cwd()
    console.log("Starting RVC...")

    input_dir = cwd / input_path
    output_dir = cwd / opt_path
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_args = [
        part
        for key, value in kwargs.items()
        if value is not None
        for part in (f"--{key}", str(value))
    ]

    process = await spawn(
        "RVC's venv python",
        await _get_rvc_executable(),
        str(cwd / "libs" / "infer_batch_rvc.py"),
        "--input_path",
        str(input_dir),
        "--opt_path",
        str(output_dir),
        "--overwrite" if overwrite else "--no-overwrite",
        *extra_args,
        cwd=os.getenv("RVC_PATH"),
    )
    result = await process.wait()

    if result != 0:
        raise SubprocessException(f"Revoicing files failed with exit code {result}")
