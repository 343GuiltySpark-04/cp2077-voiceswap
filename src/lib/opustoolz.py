import asyncio
import os
import re
from lib import ffmpeg
from util import Parallel, SubprocessException, find_files, spawn
from util.rich_console import console, create_progress


async def export_info(opusinfo_path: str, output_path: str):
    """Export opusinfo as JSON."""

    console.log("Exporting opusinfo...")

    process = await spawn(
        "OpusToolZ",
        "./libs/OpusToolZ/OpusToolZ",
        "info",
        os.path.abspath(opusinfo_path),
        os.path.abspath(output_path),
    )

    result = await process.wait()

    if result != 0:
        raise SubprocessException(
            "Exporting opusinfo failed with exit code " + str(result)
        )

    console.log("Opusinfo exported!")


async def extract_sfx(
    opusinfo_path: str, hashes: list[int], output_dir: str, to_wav=True
):
    """Extracts sfx of given hashes from the given opusinfo and opuspaks."""

    console.log("Reading SFX containers...")

    with create_progress(transient=True) as progress:
        task_id = progress.add_task("Extracting SFX", total=len(hashes), unit="file")

        process = await spawn(
            "OpusToolZ",
            "./libs/OpusToolZ/OpusToolZ",
            "extract",
            os.path.abspath(opusinfo_path),
            os.path.abspath(output_dir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )

        while not process.stdout.at_eof():
            line = b""
            try:
                line = await asyncio.wait_for(process.stdout.readline(), 1)
            except asyncio.TimeoutError:
                # keep the UI responsive while waiting for next output
                progress.refresh()
                continue

            stripped = line.decode().strip()

            if stripped.startswith("Awaiting"):
                # Give hashes
                process.stdin.write(("\n".join(map(str, hashes)) + "\n\n").encode())
                await process.stdin.drain()
            elif stripped.startswith("Wrote"):
                progress.advance(task_id)
            elif (
                stripped != ""
                and not stripped.startswith("Found")
                and not stripped.startswith("Loading")
            ):
                console.log(stripped)

        result = await process.wait()

    if result != 0:
        raise SubprocessException("Exporting SFX failed with exit code " + str(result))

    if to_wav:
        parallel = Parallel("Converting SFX to wavs")

        async def convert(file: str):
            await ffmpeg.to_wav(
                os.path.join(output_dir, file),
                os.path.join(output_dir, file.replace(".opus", ".wav")),
            )
            os.unlink(os.path.join(output_dir, file))

        for file in find_files(output_dir, ".opus"):
            parallel.run(convert, file)

        await parallel.wait()

    console.log("SFX exported!")


async def repack_sfx(opusinfo_path: str, input_dir: str, output_dir: str):
    """Repacks given folder of .wav files into paks and creates opusinfo in output path."""

    console.log("Reading SFX containers...")

    os.makedirs(output_dir, exist_ok=True)

    process = await spawn(
        "OpusToolZ",
        "./libs/OpusToolZ/OpusToolZ",
        "repack",
        os.path.abspath(opusinfo_path),
        os.path.abspath(input_dir),
        os.path.abspath(output_dir),
        stdout=asyncio.subprocess.PIPE,
    )

    await _report_repack_progress(process)
    result = await process.wait()

    if result != 0:
        raise SubprocessException(f"Repacking SFX failed with exit code {result}")

    console.log("Repacked SFX!")


async def _report_repack_progress(process):
    with create_progress(transient=True) as progress:
        packing_task_id = None
        writing_task_id = None

        while not process.stdout.at_eof():
            line = b""
            try:
                line = await asyncio.wait_for(process.stdout.readline(), 5)
            except asyncio.TimeoutError:
                # keep UI responsive while idle
                progress.refresh()
                continue

            stripped = line.decode().strip()

            if stripped.startswith("Found") and stripped.endswith("files to pack."):
                # Switch to packing phase
                if writing_task_id is not None:
                    progress.update(writing_task_id, visible=False)

                number = re.search(r"\d+", stripped).group()
                total = int(number)

                if packing_task_id is None:
                    packing_task_id = progress.add_task(
                        "Packing SFX", total=total, unit="file"
                    )
                else:
                    progress.update(
                        packing_task_id,
                        total=total,
                        completed=0,
                        description="Packing SFX",
                        unit="file",
                        visible=True,
                    )

            elif stripped.startswith("Processed file"):
                if packing_task_id is not None:
                    progress.advance(packing_task_id)

            elif stripped.startswith("Will write"):
                # Switch to writing phase
                if packing_task_id is not None:
                    progress.update(packing_task_id, visible=False)

                number = re.search(r"\d+", stripped).group()
                total = int(number)

                if writing_task_id is None:
                    writing_task_id = progress.add_task(
                        "Writing paks", total=total, unit="pak"
                    )
                else:
                    progress.update(
                        writing_task_id,
                        total=total,
                        completed=0,
                        description="Writing paks",
                        unit="pak",
                        visible=True,
                    )

            elif stripped.startswith("Wrote"):
                if writing_task_id is not None:
                    progress.advance(writing_task_id)

            elif stripped != "":
                console.log(stripped)
            else:
                # blank line heartbeat
                progress.refresh()

        # Hide any remaining tasks
        if packing_task_id is not None:
            progress.update(packing_task_id, visible=False)
        if writing_task_id is not None:
            progress.update(writing_task_id, visible=False)
