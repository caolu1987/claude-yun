"""Write transcripts as TXT and SRT."""

from pathlib import Path


def format_srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def to_txt(segments) -> str:
    lines = [seg["text"].strip() for seg in segments]
    return "\n".join(line for line in lines if line) + "\n"


def to_srt(segments) -> str:
    blocks = []
    index = 1
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        start = format_srt_time(seg["start"])
        end = format_srt_time(max(seg["end"], seg["start"]))
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
        index += 1
    return "\n".join(blocks)


def unique_stem(directory: Path, stem: str) -> str:
    """Return ``stem`` or ``stem_2``, ``stem_3``... so no existing .txt/.srt is overwritten."""
    candidate = stem
    n = 2
    while (directory / f"{candidate}.txt").exists() or (directory / f"{candidate}.srt").exists():
        candidate = f"{stem}_{n}"
        n += 1
    return candidate


def write_outputs(directory: Path, stem: str, segments) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    stem = unique_stem(directory, stem)
    txt_path = directory / f"{stem}.txt"
    srt_path = directory / f"{stem}.srt"
    # TXT gets a BOM so older Windows editors detect UTF-8; SRT stays plain UTF-8,
    # which is what subtitle tools expect.
    txt_path.write_text(to_txt(segments), encoding="utf-8-sig")
    srt_path.write_text(to_srt(segments), encoding="utf-8")
    return {"txt": txt_path, "srt": srt_path}
