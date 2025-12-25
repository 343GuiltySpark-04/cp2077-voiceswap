import json
import re
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from util import find_files
from util.rich_console import console, create_progress


GENDERS = ("female", "male")
VOICEOVER_PREFIX = "voiceovermap"
JSON_SUFFIX = ".json.json"
DATA_ENTRY_KEYS = ("Data", "RootChunk", "root", "Data", "entries")
_SANITIZE_PATTERN = re.compile(r".$")
_WORD_PATTERN = re.compile(r"\w+")


def _strip_suffix(value: str, suffix: str) -> str:
    if value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def _normalize_subtitle_path(relative_path: str, skip_token: str) -> str:
    normalized = relative_path.replace("\\", "/")
    normalized = _strip_suffix(normalized, JSON_SUFFIX)
    return normalized.replace(skip_token, "{}")


def _normalize_depot_path(depot_path: str, skip_token: str) -> str:
    normalized = depot_path.replace("\\", "/")
    return normalized.replace(skip_token, "{}")


def _load_entries(root: Path, relative_path: str) -> List[Dict[str, Any]]:
    file_path = root / Path(relative_path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    entries: Any = data
    for key in DATA_ENTRY_KEYS:
        entries = entries[key]
    return entries


def _extract_effect_type(relative_path: str) -> str:
    filename = Path(relative_path).name
    stem = filename.split(".", 2)[0]
    effect = stem.split("_")[-1]
    return "main" if effect in ("1", VOICEOVER_PREFIX) else effect


def _is_voiceovermap_file(relative_path: str) -> bool:
    filename = Path(relative_path).name
    return filename.startswith(VOICEOVER_PREFIX) and filename.endswith(JSON_SUFFIX)


def _is_subtitle_candidate(relative_path: str) -> bool:
    filename = Path(relative_path).name
    if not filename.endswith(JSON_SUFFIX):
        return False
    return not filename.startswith((VOICEOVER_PREFIX, "subtitles"))


def _build_subtitle_item(
    entry: Dict[str, Any],
    relative_path: str,
    skip_token: str,
) -> Optional[Dict[str, Any]]:
    item: Dict[str, Any] = {}
    for gender in GENDERS:
        text = entry.get(f"{gender}Variant")
        if text:
            item[gender] = {"text": text}

    if not item:
        return None

    item["_path"] = _normalize_subtitle_path(relative_path, skip_token)
    return item


def map_subtitles(path: str, locale: str) -> Dict[str, Dict[str, Any]]:
    """Map subtitle entries to their corresponding voiceover audio files."""

    root = Path(path)
    skip_token = f"localization/{locale}"

    subtitle_files = [
        rel_path
        for rel_path in find_files(path, JSON_SUFFIX, locale)
        if _is_subtitle_candidate(rel_path)
    ]
    map_files = [
        rel_path
        for rel_path in chain(
            find_files(path, subfolder=locale),
            find_files(path, subfolder="en-us"),
            find_files(path, subfolder="common"),
        )
        if _is_voiceovermap_file(rel_path)
    ]

    vo_map: Dict[str, Dict[str, Any]] = {}
    found_files: set[str] = set()
    not_found_subtitles: set[str] = set()

    with create_progress() as progress:
        scan_task = (
            progress.add_task(
                "Scanning subtitle files",
                total=len(subtitle_files),
                unit="file",
            )
            if subtitle_files
            else None
        )

        for relative_path in subtitle_files:
            entries = _load_entries(root, relative_path)
            for entry in entries:
                item = _build_subtitle_item(entry, relative_path, skip_token)
                if item:
                    vo_map[entry["stringId"]] = item
            if scan_task is not None:
                progress.advance(scan_task)

        map_task = (
            progress.add_task(
                "Searching for mappings",
                total=len(map_files),
                unit="file",
            )
            if map_files
            else None
        )

        for relative_path in map_files:
            effect_type = _extract_effect_type(relative_path)
            entries = _load_entries(root, relative_path)

            for entry in entries:
                string_id = entry["stringId"]
                if string_id not in vo_map:
                    not_found_subtitles.add(string_id)
                    vo_map[string_id] = {gender: {"vo": {}} for gender in GENDERS}

                item = vo_map[string_id]
                for gender in GENDERS:
                    item.setdefault(gender, {"vo": {}})

                found_files.add(string_id)
                prev_depot: Optional[str] = None

                for gender in list(item.keys()):
                    if gender.startswith("_"):
                        continue

                    subitem = item[gender]
                    depot_info = entry.get(f"{gender}ResPath")
                    if not depot_info:
                        continue

                    depot_path = depot_info["DepotPath"]["$value"]

                    if prev_depot and depot_path == prev_depot:
                        del item[gender]
                        break

                    prev_depot = depot_path

                    subitem.setdefault("vo", {})
                    subitem["vo"][effect_type] = _normalize_depot_path(
                        depot_path, skip_token
                    )
            if map_task is not None:
                progress.advance(map_task)

    console.log(f"Found {len(found_files)} subtitle mappings.")
    missing_vo = len(vo_map) - len(found_files)
    if missing_vo > 0:
        console.log(
            f"[yellow]Warning: Not found voiceovers for {missing_vo} subtitles![/]"
        )

    missing_sub = len(not_found_subtitles)
    if missing_sub > 0:
        console.log(
            f"[yellow]Warning: Not found subtitles for {missing_sub} voiceovers![/]"
        )

    return vo_map


_g_tts: Any = None
_g_reference: Optional[str] = None
_g_language: Optional[str] = None


def _init_tts_worker(reference: str, language: str) -> None:
    """Initialize the TTS engine inside a worker process."""
    global _g_tts, _g_reference, _g_language
    _g_reference = reference
    _g_language = language

    from TTS.api import TTS  # type: ignore

    _g_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")


def _sanitize_text(text: str) -> str:
    sanitized = _SANITIZE_PATTERN.sub("", text)
    return sanitized.replace(".", ",-;")


def _tts_worker(entry: Sequence[str]) -> None:
    if not entry:
        return

    file_path = entry[0]
    text = entry[1]
    reference = entry[2] if len(entry) > 2 else _g_reference

    if reference is None or _g_tts is None or _g_language is None:
        raise RuntimeError("TTS worker was not initialized correctly.")

    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sanitized_text = _sanitize_text(text)
    if not sanitized_text or not _WORD_PATTERN.match(sanitized_text):
        return

    _g_tts.tts_to_file(
        text=sanitized_text,
        speaker_wav=reference,
        language=_g_language,
        file_path=file_path,
    )


def generate_speech(
    files: List[List[str]],
    reference: str,
    language: str,
    batchsize: int = 1,
) -> None:
    """Generate text-to-speech audio for the provided files."""
    if not files:
        console.log("No speech lines to synthesize.")
        return

    console.log("Loading TTS...")

    with create_progress() as progress:
        task = progress.add_task(
            "Generating speech",
            total=len(files),
            unit="line",
        )
        with Pool(max(batchsize, 1), _init_tts_worker, (reference, language)) as pool:
            for _ in pool.imap_unordered(_tts_worker, files):
                progress.advance(task)
