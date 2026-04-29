#!/usr/bin/env python3
"""
Arabic ASR experiment runner for Whisper / whisper.cpp outputs.

This script makes the ASR improvement aspect explicit:
- baseline Whisper transcription before fine-tuning
- improved model transcription after fine-tuning
- Arabic normalization
- long-form chunking and reconstruction
- context continuity across chunks
- transcript post-processing
- WER/CER evaluation

Manifest format, one JSON object per line:
{"audio": "path/to/audio_or_video.mp4", "reference": "gold Arabic transcript", "id": "optional_id"}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


AR_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
PUNCT_RE = re.compile(r"[^\w\s\u0600-\u06FF]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] Command failed:", " ".join(cmd), file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result.stdout.strip()


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for fair WER/CER evaluation."""
    text = text.strip()
    text = AR_DIACRITICS_RE.sub("", text)
    text = text.replace("ـ", "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = text.replace("ة", "ه")
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    ref_chars = reference.replace(" ", "")
    hyp_chars = hypothesis.replace(" ", "")
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def audio_duration_seconds(path: Path) -> float:
    out = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    return float(out)


def extract_wav(input_path: Path, wav_path: Path, start: float | None = None, duration: float | None = None) -> None:
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd.extend(["-ss", f"{start:.3f}"])
    cmd.extend(["-i", str(input_path)])
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])
    cmd.extend(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path), "-loglevel", "error"])
    run(cmd)


def ts_to_seconds(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_whisper_json(json_path: Path, offset: float = 0.0) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    segments = []
    for seg in data.get("transcription", []):
        text = seg.get("text", "").strip()
        timestamps = seg.get("timestamps", {})
        if not text:
            continue
        try:
            start = ts_to_seconds(timestamps["from"]) + offset
            end = ts_to_seconds(timestamps["to"]) + offset
        except Exception:
            start = offset
            end = offset
        segments.append({"start_sec": start, "end_sec": end, "text": text})
    return segments


@dataclass
class WhisperCppConfig:
    whisper_bin: str
    model: Path
    language: str = "ar"
    threads: int = 8
    chunk_seconds: float = 30.0
    overlap_seconds: float = 2.0
    context_words: int = 24
    use_prompt_context: bool = True


def transcript_context(text: str, max_words: int) -> str:
    words = text.split()
    if max_words <= 0 or not words:
        return ""
    return " ".join(words[-max_words:])


def postprocess_transcript(text: str) -> str:
    """Clean merged transcript spacing and apply light Arabic punctuation correction."""
    text = SPACE_RE.sub(" ", text).strip()
    text = re.sub(r"\s+([،؛؟,.!?])", r"\1", text)
    text = re.sub(r"([،؛؟,.!?])([^\s])", r"\1 \2", text)
    text = re.sub(r"([.؟!])\s+\1+", r"\1 ", text)
    text = SPACE_RE.sub(" ", text).strip()
    if text and text[-1] not in ".؟!":
        text += "."
    return text


def deduplicate_overlap(prev_text: str, new_text: str, max_overlap_words: int = 18) -> str:
    """Remove repeated words caused by overlapping audio chunks."""
    prev_words = prev_text.split()
    new_words = new_text.split()
    max_k = min(max_overlap_words, len(prev_words), len(new_words))
    for k in range(max_k, 0, -1):
        if prev_words[-k:] == new_words[:k]:
            return " ".join(new_words[k:])
    return new_text


def transcribe_chunk(wav_path: Path, out_stem: Path, cfg: WhisperCppConfig, prompt: str = "") -> list[dict]:
    cmd = [
        cfg.whisper_bin,
        "-m", str(cfg.model),
        "-f", str(wav_path),
        "--language", cfg.language,
        "-ojf",
        "-of", str(out_stem),
        "-t", str(cfg.threads),
    ]
    if cfg.use_prompt_context and prompt:
        cmd.extend(["--prompt", prompt])
    run(cmd)
    json_path = out_stem.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"whisper.cpp did not produce {json_path}")
    return parse_whisper_json(json_path)


def transcribe_long_form(audio_path: Path, cfg: WhisperCppConfig) -> tuple[str, list[dict]]:
    """Chunk long audio, carry context between chunks, and reconstruct ordered transcript."""
    duration = audio_duration_seconds(audio_path)
    segments: list[dict] = []
    merged_text = ""

    with tempfile.TemporaryDirectory(prefix="asr_chunks_") as tmp:
        tmp_dir = Path(tmp)
        start = 0.0
        chunk_idx = 0
        while start < duration:
            chunk_idx += 1
            chunk_duration = min(cfg.chunk_seconds, duration - start)
            wav_path = tmp_dir / f"chunk_{chunk_idx:04d}.wav"
            out_stem = tmp_dir / f"chunk_{chunk_idx:04d}"

            extract_wav(audio_path, wav_path, start=start, duration=chunk_duration)
            prompt = transcript_context(merged_text, cfg.context_words)
            chunk_segments = transcribe_chunk(wav_path, out_stem, cfg, prompt=prompt)

            keep_start = start if chunk_idx == 1 else start + cfg.overlap_seconds
            kept_text_parts = []
            for seg in chunk_segments:
                shifted = {
                    "start_sec": seg["start_sec"] + start,
                    "end_sec": seg["end_sec"] + start,
                    "text": seg["text"],
                }
                if shifted["end_sec"] > keep_start:
                    segments.append(shifted)
                    kept_text_parts.append(shifted["text"].strip())

            chunk_text = SPACE_RE.sub(" ", " ".join(kept_text_parts)).strip()
            if chunk_text:
                chunk_text = deduplicate_overlap(merged_text, chunk_text)
                merged_text = SPACE_RE.sub(" ", f"{merged_text} {chunk_text}").strip()

            step = cfg.chunk_seconds - cfg.overlap_seconds
            if step <= 0:
                raise ValueError("chunk_seconds must be larger than overlap_seconds")
            start += step

    segments.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
    transcript = postprocess_transcript(merged_text)
    return transcript, segments


def command_transcribe(args: argparse.Namespace) -> None:
    manifest = read_jsonl(Path(args.manifest))
    cfg = WhisperCppConfig(
        whisper_bin=args.whisper_bin,
        model=Path(args.model).expanduser(),
        language=args.language,
        threads=args.threads,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        context_words=args.context_words,
        use_prompt_context=not args.no_prompt_context,
    )

    rows = []
    for idx, item in enumerate(manifest, start=1):
        audio = Path(item["audio"]).expanduser()
        item_id = item.get("id") or audio.stem
        print(f"[{idx}/{len(manifest)}] {args.system_name}: {item_id}")
        hypothesis, segments = transcribe_long_form(audio, cfg)
        rows.append({
            "id": item_id,
            "audio": str(audio),
            "system": args.system_name,
            "reference": item.get("reference", ""),
            "hypothesis": hypothesis,
            "segments": segments,
        })

    write_jsonl(Path(args.out), rows)
    print(f"[OK] wrote predictions: {args.out}")


def command_evaluate(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.pred))
    details = []
    sum_wer_raw = 0.0
    sum_wer_norm = 0.0
    sum_cer_norm = 0.0

    for row in rows:
        ref = row.get("reference", "")
        hyp = row.get("hypothesis", "")
        ref_norm = normalize_arabic(ref)
        hyp_norm = normalize_arabic(hyp)

        metrics = {
            "id": row.get("id", ""),
            "system": row.get("system", ""),
            "wer_raw": wer(ref, hyp),
            "wer_normalized": wer(ref_norm, hyp_norm),
            "cer_normalized": cer(ref_norm, hyp_norm),
            "reference_normalized": ref_norm,
            "hypothesis_normalized": hyp_norm,
        }
        details.append(metrics)
        sum_wer_raw += metrics["wer_raw"]
        sum_wer_norm += metrics["wer_normalized"]
        sum_cer_norm += metrics["cer_normalized"]

    n = len(details) or 1
    summary = {
        "prediction_file": args.pred,
        "num_files": len(details),
        "avg_wer_raw": sum_wer_raw / n,
        "avg_wer_normalized": sum_wer_norm / n,
        "avg_cer_normalized": sum_cer_norm / n,
        "details": details,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, ensure_ascii=False, indent=2))
    print(f"[OK] wrote metrics: {out}")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "system", "wer_raw", "wer_normalized", "cer_normalized"])
            writer.writeheader()
            for row in details:
                writer.writerow({k: row[k] for k in writer.fieldnames})
        print(f"[OK] wrote CSV: {csv_path}")


def command_make_manifest(args: argparse.Namespace) -> None:
    """Create a starter manifest from matching audio/txt files."""
    audio_dir = Path(args.audio_dir).expanduser()
    ref_dir = Path(args.ref_dir).expanduser()
    rows = []
    for audio in sorted(audio_dir.glob(args.audio_glob)):
        ref_path = ref_dir / f"{audio.stem}.txt"
        if not ref_path.exists():
            print(f"[WARN] missing reference for {audio.name}: {ref_path}")
            continue
        rows.append({
            "id": audio.stem,
            "audio": str(audio),
            "reference": ref_path.read_text(encoding="utf-8").strip(),
        })
    write_jsonl(Path(args.out), rows)
    print(f"[OK] wrote manifest with {len(rows)} items: {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arabic ASR baseline/improvement experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    mk = sub.add_parser("make-manifest", help="build JSONL manifest from audio files and .txt references")
    mk.add_argument("--audio-dir", required=True)
    mk.add_argument("--ref-dir", required=True)
    mk.add_argument("--audio-glob", default="*.wav")
    mk.add_argument("--out", required=True)
    mk.set_defaults(func=command_make_manifest)

    tr = sub.add_parser("transcribe", help="transcribe manifest with whisper.cpp and long-form chunking")
    tr.add_argument("--manifest", required=True)
    tr.add_argument("--model", required=True)
    tr.add_argument("--system-name", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--whisper-bin", default="whisper-cli")
    tr.add_argument("--language", default="ar")
    tr.add_argument("--threads", type=int, default=8)
    tr.add_argument("--chunk-seconds", type=float, default=30.0)
    tr.add_argument("--overlap-seconds", type=float, default=2.0)
    tr.add_argument("--context-words", type=int, default=24)
    tr.add_argument("--no-prompt-context", action="store_true")
    tr.set_defaults(func=command_transcribe)

    ev = sub.add_parser("evaluate", help="compute WER/CER for predictions JSONL")
    ev.add_argument("--pred", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--csv")
    ev.set_defaults(func=command_evaluate)

    return parser


def main() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("[ERROR] ffmpeg and ffprobe are required.", file=sys.stderr)
        raise SystemExit(1)
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
