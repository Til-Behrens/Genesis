"""Turn the captioned clip metadata into the dataset file DiffSynth's train.py reads."""
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from genesis import events
from genesis.events import Event

logger = logging.getLogger(__name__)

PASSTHROUGH_KEYS = ("audio_text", "original_video", "start_time")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one json object per non-empty line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows atomically, so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def to_diffsynth_row(record: dict[str, Any]) -> dict[str, Any]:
    """Map a metadata row onto the `video` and `prompt` keys read by DiffSynth's train.py.

    Raises:
        ValueError: The row names no video file.
    """
    video = record.get("video") or record.get("file_name")
    if not video:
        raise ValueError("row has neither 'video' nor 'file_name'")
    row = {"video": str(video), "prompt": str(record.get("prompt") or "").strip()}
    row.update({key: record[key] for key in PASSTHROUGH_KEYS if key in record})
    return row


def prepare_dataset(
    clips_dir: Path | str,
    metadata_path: Path | str,
    output_path: Path | str,
) -> Iterator[Event]:
    """Validate clips and captions and write the training metadata.

    Rows whose clip is missing or whose prompt is empty are left out, so failed
    captions never reach training.

    Yields:
        One `progress` event per skipped row, then `complete` (or `error`).
    """
    clips_dir, metadata_path, output_path = Path(clips_dir), Path(metadata_path), Path(output_path)
    if not metadata_path.exists():
        yield events.error(f"Metadata file not found: {metadata_path}")
        return

    records = read_jsonl(metadata_path)
    if not records:
        yield events.error("Metadata file is empty.")
        return

    rows: list[dict[str, Any]] = []
    skipped = {"invalid": 0, "missing video": 0, "no caption": 0}

    for index, record in enumerate(records, 1):
        try:
            row = to_diffsynth_row(record)
        except ValueError as e:
            reason, name = "invalid", f"row {index}: {e}"
        else:
            name = row["video"]
            if not (clips_dir / row["video"]).exists():
                reason = "missing video"
            elif not row["prompt"]:
                reason = "no caption"
            else:
                rows.append(row)
                continue
        skipped[reason] += 1
        yield events.progress(index, len(records), f"{name}: {reason}, skipped")

    if not rows:
        yield events.error("No usable rows: every clip is missing or uncaptioned.")
        return

    write_jsonl(output_path, rows)
    summary = ", ".join(f"{count} {reason}" for reason, count in skipped.items() if count) or "none"
    yield events.complete(
        f"{len(rows)} of {len(records)} clips ready, skipped: {summary}\nTraining metadata: {output_path}",
        rows=len(rows),
        skipped=skipped,
        output_path=str(output_path),
    )
