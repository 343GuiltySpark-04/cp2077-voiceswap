import asyncio
import os
import re
from threading import Thread

import nest_asyncio
from waapi import CannotConnectToWaapiException, WaapiClient, WaapiRequestFailed

from util import SubprocessException, spawn, watch_async
from util.rich_console import console, create_progress


nest_asyncio.apply()  # needed for waapi

WWISE_OBJECT_PATH = "\\Actor-Mixer Hierarchy\\Default Work Unit\\"


def _get_wwise_path():
    """Returns the path to the Wwise executable."""

    path = os.getenv("WWISEROOT") or os.getenv("WWWISE_PATH")
    if not path:
        raise RuntimeError(
            "Neither 'WWISEROOT' nor 'WWWISE_PATH' environment variable is set"
        )

    return os.path.join(path, "Authoring/x64/Release/bin/")


def _get_wwise_project(folder: str):
    basename = os.path.basename(folder.replace(r"[\/]$", ""))

    return os.path.join(folder, basename + ".wproj")


async def create_project(project_dir: str):
    """Creates a new Wwise project at the given path if it doesnt exist"""

    project_path = _get_wwise_project(project_dir)

    # Check existence
    if os.path.exists(project_path):
        return

    # Otherwise, create the project
    process = await spawn(
        "Wwise",
        _get_wwise_path() + "WwiseConsole",
        "create-new-project",
        project_path,
        "--quiet",
    )
    result = await process.wait()

    if result != 0 and result != 2:  # 2 means warnings in wwise
        raise SubprocessException(f"Creating project failed with exit code {result}")

    # Find desired quality settings
    conv_id = ""
    with open(
        os.path.join(
            project_dir, "Conversion Settings/Factory Conversion Settings.wwu"
        ),
        "r",
        encoding="utf-8",
    ) as f:
        settings = f.read()
        findings = re.search(
            r"<Conversion Name=\"Vorbis Quality High\" ID=\"{([\w\-]+)}\">", settings
        )
        conv_id = findings.group(1)

    # Patch project settings
    data = ""
    with open(project_path, "r", encoding="utf-8") as f:
        data = f.read()

    data = re.sub(
        r"<DefaultConversion(.*)ID=\"\{([\w\-]+)\}\"\/>\n",
        r'<DefaultConversion\g<1>ID="{' + conv_id + '}"/>\n',
        data,
    )
    data = data.replace("Default Conversion Settings", "Vorbis Quality High")

    with open(project_path, "w", encoding="utf-8") as f:
        f.write(data)


async def spawn_wwise(project_dir: str):
    """Starts Wwise for the given project."""

    process = await spawn(
        "Wwise",
        _get_wwise_path() + "Wwise",
        _get_wwise_project(project_dir),
        "--quiet",
    )
    yield process

    result = await process.wait()
    if result != 0:
        raise SubprocessException(f"Wwise server failed with exit code {result}")


async def _create_waapi(server):
    waapi = None
    while server.returncode is None:
        try:
            waapi = WaapiClient(allow_exception=True)
            break
        except CannotConnectToWaapiException as err:
            if server.returncode is not None:
                raise err

    if waapi is None:
        raise CannotConnectToWaapiException()

    return waapi


async def wait_waapi_load(waapi: WaapiClient):
    """Wait until the WAAPI server is loaded."""

    console.log("Waiting for Wwise to load...")

    # The 'loaded' event was unreliable

    while True:
        try:
            result = waapi.call("ak.wwise.core.getInfo")
            if result:
                return
        except WaapiRequestFailed:
            pass
        await asyncio.sleep(0.5)


def move_wwise_files(converted_objects: list[dict], output_path: str):
    """Moves all converted Wwise files to the given output folder and renames them correctly."""

    sounds = [item for item in converted_objects if item["type"] == "Sound"]
    if not sounds:
        return

    with create_progress(transient=True) as progress:
        task_id = progress.add_task(
            "Renaming files",
            total=len(sounds),
            unit="file",
        )

        for obj in sounds:
            original_path = obj["sound:originalWavFilePath"]
            filename = os.path.basename(original_path)[: -len(".wav")]

            path = obj["path"][len(WWISE_OBJECT_PATH) : -len(obj["name"])]
            output_dir = os.path.join(output_path, path)
            output = os.path.join(output_dir, filename + ".wem")

            os.makedirs(output_dir, exist_ok=True)

            if os.path.exists(output):
                os.unlink(output)

            os.rename(obj["sound:convertedWemFilePath"], output)
            progress.advance(task_id)


def move_wwise_files_auto(project_dir: str, output_path: str):
    """
    Tries to find the correct output path for all converted files

    without the index of converted files.
    """

    cache_dir = os.path.join(project_dir, ".cache/Windows/SFX")

    found_files = []

    for root, _dirs, files in os.walk(cache_dir):
        path = root[len(cache_dir) + 1 :]

        for file in files:
            if not file.endswith(".wem"):
                continue

            found_files.append(os.path.join(path, file))

    if not found_files:
        return

    with create_progress(transient=True) as progress:
        task_id = progress.add_task("Moving files", total=len(found_files), unit="file")

        for file in found_files:
            new_path = os.path.join(output_path, file.replace("_3F75BDB9", ""))
            new_dir = os.path.dirname(new_path)

            os.makedirs(new_dir, exist_ok=True)
            if os.path.exists(new_path):
                os.unlink(new_path)

            os.rename(os.path.join(cache_dir, file), new_path)
            progress.advance(task_id)


async def _convert_files(
    input_path: str, project_dir: str, output_path: str, override: bool, waapi
):
    # Wait for load
    await wait_waapi_load(waapi)

    console.log("Starting automation mode...")

    # Setup
    waapi.call("ak.wwise.debug.enableAutomationMode", {"enable": True})

    # List all files
    console.log("Starting import...")
    to_import = []
    skipped = 0

    for root, _dirs, files in os.walk(input_path):
        relative_root = root[len(input_path) + 1 :]
        path = "\\".join("<Folder>" + s for s in re.split(r"[\\/]", relative_root) if s)

        for file in files:

            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(relative_root, file)
            if not override and os.path.exists(
                os.path.join(output_path, re.sub(r"\.wav$", ".wem", file_path))
            ):
                skipped += 1
                continue

            to_import.append(
                {
                    "audioFile": os.path.abspath(os.path.join(input_path, file_path)),
                    "originalsSubFolder": relative_root,
                    "objectPath": os.path.join(
                        WWISE_OBJECT_PATH, path, "<Sound>" + file
                    ),
                }
            )

    if skipped > 0:
        console.log(f"Not overwriting {skipped} already processed files")

    imported_ids: list[str] = []
    handler = None
    with create_progress(transient=True) as progress:
        import_task_id = progress.add_task(
            "Importing files to Wwise",
            total=len(to_import),
            unit="file",
        )

        def on_object_created(*_args, **kwargs):
            if kwargs["object"]["type"] == "AudioFileSource":
                progress.advance(import_task_id)

        handler = waapi.subscribe(
            "ak.wwise.core.object.created", on_object_created, {"return": ["type"]}
        )

        try:
            imported_objects = waapi.call(
                "ak.wwise.core.audio.import",
                {
                    "importOperation": "replaceExisting",
                    "imports": to_import,
                },
            )["objects"]
        finally:
            if handler is not None:
                handler.unsubscribe()
                handler = None

        imported_ids = [
            obj["id"] for obj in imported_objects if obj["name"].endswith("_wav")
        ]
        progress.update(import_task_id, completed=len(to_import))

        # Create convert thread
    console.log("Starting conversion...")
    convert_thread = Thread(
        target=waapi.call,
        args=(
            "ak.wwise.ui.commands.execute",
            {
                "command": "ConvertAllPlatform",
                "objects": imported_ids,
            },
        ),
    )

    # Listen for conversion completion
    converted_objects = []

    def on_converted(command: str, objects, *_args, **_kwargs):
        nonlocal converted_objects
        if command == "ConvertAllPlatform":
            converted_objects = objects

    handler = waapi.subscribe(
        "ak.wwise.ui.commands.executed",
        on_converted,
        {
            "return": [
                "type",
                "path",
                "name",
                "sound:originalWavFilePath",
                "sound:convertedWemFilePath",
            ]
        },
    )

    cache_dir = os.path.join(project_dir, ".cache")

    os.makedirs(cache_dir, exist_ok=True)
    event_queue, observer = watch_async(cache_dir, recursive=True)

    total_imports = len(imported_ids)
    completed_imports = 0

    convert_thread.start()
    try:
        with create_progress(transient=True) as progress:
            convert_task_id = progress.add_task(
                "Converting files",
                total=total_imports,
                unit="file",
            )

            while convert_thread.is_alive():
                try:
                    event = event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    progress.refresh()
                    convert_thread.join(0.1)
                else:
                    if (
                        event.src_path.endswith(".wem")
                        and completed_imports < total_imports
                    ):
                        completed_imports += 1
                        progress.advance(convert_task_id)

                if total_imports > 0 and completed_imports >= total_imports:
                    convert_thread.join(60)
                    break

            while True:
                try:
                    event = event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    if (
                        event.src_path.endswith(".wem")
                        and completed_imports < total_imports
                    ):
                        completed_imports += 1
                        progress.advance(convert_task_id)

            progress.update(
                convert_task_id,
                completed=min(completed_imports, total_imports),
            )
    finally:
        observer.stop()
        if convert_thread.is_alive():
            convert_thread.join(timeout=0)

    # handler.unsubscribe() # Prevents the code from continuing

    # Move and rename files
    console.log("Starting moving files...")
    move_wwise_files(converted_objects, output_path)

    console.log("Conversion done!")


async def convert_files(
    input_path: str, project_dir: str, output_path: str, override: bool
):
    """Converts all files in the given folder to Wwise format."""
    await create_project(project_dir)

    console.print("##############################################################")
    console.print("# Opening Wwise in a few seconds, get ready for a jumpscare! #")
    console.print("#  The Wwise window will be fully automated, don't touch it! #")
    console.print("##############################################################")

    await asyncio.sleep(5)

    # Start the WAAPI server
    server_spawn = spawn_wwise(project_dir)

    server = await anext(server_spawn)

    # Try to estabilish connection
    console.log("Trying to connect to Wwise...")
    waapi = await _create_waapi(server)
    console.log("Connected!")

    # Run the script
    try:
        await _convert_files(input_path, project_dir, output_path, override, waapi)
    finally:
        # Close the WAAPI server
        if server.returncode is None:
            console.log("Closing Wwise...")
            server.terminate()
            await server.wait()
        waapi.disconnect()
