#!/usr/bin/env python3
"""
Islamic Podcast Clipper
whisper.cpp → transcript JSON
MALIK       → clip timestamps JSON
clipper.py  → cuts, captions, burns

Note:
This script is the downstream clipping/agent component. For the main ASR
improvement experiments required by the project feedback, use
asr_experiments.py. That script defines the Whisper baseline, Arabic text
normalization, long-form chunking/reconstruction, and WER/CER evaluation.
"""

import os
import sys
import json
import subprocess
import re
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WHISPER_BIN   = "whisper-cli"
WHISPER_MODEL = os.path.expanduser("~/Documents/whisper-models/ggml-large-v3-turbo.bin")
# ─────────────────────────────────────────────


def run(cmd, **kwargs):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"[ERROR] Command failed:\n{cmd}\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


# ──────────────────────────────────────────────────────────
# 1. TRANSCRIPTION  (whisper.cpp)
# ──────────────────────────────────────────────────────────

def transcribe(video_path: Path, out_dir: Path) -> Path:
    """Extract audio, run whisper.cpp, return path to JSON output."""
    json_path = out_dir / f"{video_path.stem}.json"
    if json_path.exists():
        print(f"[SKIP] Transcript already exists: {json_path}")
        return json_path

    wav_path = out_dir / f"{video_path.stem}.wav"
    if not wav_path.exists():
        print(f"[1/2] Extracting audio...")
        run(f'ffmpeg -y -i "{video_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{wav_path}" -loglevel error')

    print(f"[1/2] Transcribing with whisper.cpp...")
    out_stem = out_dir / video_path.stem
    run(
        f'{WHISPER_BIN} -m "{WHISPER_MODEL}" -f "{wav_path}" '
        f'--language ar '
        f'-ojf '
        f'-of "{out_stem}" '
        f'-t 8'
    )

    if not json_path.exists():
        print(f"[ERROR] whisper.cpp did not produce {json_path}")
        sys.exit(1)
    print(f"[OK] Transcript saved: {json_path}")
    return json_path


# ──────────────────────────────────────────────────────────
# 2. PARSE whisper JSON
# ──────────────────────────────────────────────────────────

def _ts_to_secs(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_whisper_json(json_path: Path) -> list:
    """Returns list of {start_sec, end_sec, text} from whisper.cpp JSON."""
    data    = json.loads(json_path.read_bytes().decode("utf-8", errors="ignore"))
    entries = []
    for seg in data.get("transcription", []):
        text = seg.get("text", "").strip()
        t    = seg.get("timestamps", {})
        if not text:
            continue
        try:
            entries.append({
                "start_sec": _ts_to_secs(t["from"]),
                "end_sec":   _ts_to_secs(t["to"]),
                "text":      text,
            })
        except Exception:
            continue
    return entries


# ──────────────────────────────────────────────────────────
# 3. CUT
# ──────────────────────────────────────────────────────────

def _secs_to_ts(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def cut_clip(video_path: Path, start: float, end: float,
             clip_dir: Path, clip_name: str) -> Path:
    out_mp4 = clip_dir / f"{clip_name}_raw.mp4"
    if out_mp4.exists():
        print(f"  [SKIP] Already cut: {out_mp4.name}")
        return out_mp4
    duration = end - start
    run(f'ffmpeg -y -i "{video_path}" -ss {_secs_to_ts(start)} -t {duration:.3f} '
        f'-c:v libx264 -c:a aac "{out_mp4}" -loglevel error')
    return out_mp4


# ──────────────────────────────────────────────────────────
# 4. VERTICAL REFORMAT
# ──────────────────────────────────────────────────────────

def make_vertical(raw_mp4: Path, clip_dir: Path, clip_name: str) -> Path:
    vertical_mp4 = clip_dir / f"{clip_name}_vertical.mp4"
    if vertical_mp4.exists():
        print(f"  [SKIP] Already vertical: {vertical_mp4.name}")
        return vertical_mp4
    filtergraph = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:20[bg];"
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2"
    )
    run(f'ffmpeg -y -i "{raw_mp4}" -filter_complex "{filtergraph}" '
        f'-c:a copy "{vertical_mp4}" -loglevel error')
    return vertical_mp4


# ──────────────────────────────────────────────────────────
# 5. CAPTION
# ──────────────────────────────────────────────────────────

def entries_to_srt(entries: list, start_offset: float) -> str:
    def _fmt(sec: float) -> str:
        sec = max(0.0, sec)
        h   = int(sec // 3600)
        m   = int((sec % 3600) // 60)
        s2  = int(sec % 60)
        ms  = int(round((sec % 1) * 1000))
        return f"{h:02d}:{m:02d}:{s2:02d},{ms:03d}"

    lines = []
    for idx, e in enumerate(entries, 1):
        s = e["start_sec"] - start_offset
        e_ = e["end_sec"]  - start_offset
        lines.append(f"{idx}\n{_fmt(s)} --> {_fmt(e_)}\n{e['text']}\n")
    return "\n".join(lines)


_ASS_HEADER = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 1280\n"
    "PlayResY: 720\n"
    "ScaledBorderAndShadow: yes\n"
    "\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding\n"
    "Style: Default,Damascus,18,&Hffffff,&Hffffff,&H0,&H0,1,0,0,0,100,100,0,0,1,1,0,2,20,240,250,1\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)


def _srt_ts_to_ass(t: str) -> str:
    t = t.strip().replace(",", ".")
    h, m, s = t.split(":")
    return f"{int(h)}:{m}:{float(s):05.2f}"


def srt_to_ass(srt_path: Path, ass_path: Path):
    blocks = re.split(r'\n\n+', srt_path.read_text(encoding="utf-8").strip())
    events = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            start_str, end_str = lines[1].split(" --> ")
            text = " ".join(lines[2:])
            events.append(
                f"Dialogue: 0,{_srt_ts_to_ass(start_str)},{_srt_ts_to_ass(end_str)},"
                f"Default,,0,0,0,,{text}"
            )
        except Exception:
            continue
    ass_path.write_text(_ASS_HEADER + "\n".join(events) + "\n", encoding="utf-8")


def burn_captions(raw_mp4: Path, ass_path: Path, final_mp4: Path):
    filter_complex = (
        "[0:v]drawbox=x=0:y=1140:w=iw:h=140:color=black:t=fill[blacked];"
        f"[blacked]ass={ass_path}"
    )
    run(f'ffmpeg -y -i "{raw_mp4}" -filter_complex "{filter_complex}" '
        f'-c:a copy "{final_mp4}" -loglevel error')


# ──────────────────────────────────────────────────────────
# 6. REBURN
# ──────────────────────────────────────────────────────────

def reburn(clip_dir: Path):
    clip_name    = clip_dir.name
    vertical_mp4 = clip_dir / f"{clip_name}_vertical.mp4"
    raw_mp4      = clip_dir / f"{clip_name}_raw.mp4"
    ass_path     = clip_dir / f"{clip_name}.ass"
    final_mp4    = clip_dir / f"{clip_name}.mp4"
    source_mp4   = vertical_mp4 if vertical_mp4.exists() else raw_mp4
    if not source_mp4.exists() or not ass_path.exists():
        print(f"[ERROR] Missing files in {clip_dir}")
        sys.exit(1)
    print(f"[REBURN] {ass_path.name} → {final_mp4.name} (source: {source_mp4.name})")
    burn_captions(source_mp4, ass_path, final_mp4)
    print("[OK] Done.")


# ──────────────────────────────────────────────────────────
# TEST MODE
# ──────────────────────────────────────────────────────────

def run_test(video_path: Path):
    """Cut first 90s, generate dummy captions, burn — fast font/position check."""
    out_dir  = video_path.parent / f"{video_path.stem}_clips"
    out_dir.mkdir(exist_ok=True)
    clip_dir = out_dir / "clip_test"
    clip_dir.mkdir(exist_ok=True)

    print(f"\n[TEST] Cutting first 90s of {video_path.name}...")
    raw_mp4 = cut_clip(video_path, 0, 90, clip_dir, "clip_test")

    # Generate sample Arabic captions spread across 90s
    samples = [
        (0,   5,  "بسم الله الرحمن الرحيم"),
        (5,   15, "هذا اختبار لحجم الخط وموضع الترجمة"),
        (15,  30, "الإمام مسلم بن الحجاج رحمه الله"),
        (30,  50, "قال النبي صلى الله عليه وسلم: خيركم من تعلم القرآن وعلمه"),
        (50,  70, "العلم النافع هو علم القلب لا علم اللسان فقط"),
        (70,  90, "جزاكم الله خيراً — نهاية الاختبار"),
    ]

    def _fmt(sec):
        h  = int(sec // 3600)
        m  = int((sec % 3600) // 60)
        s2 = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s2:02d},000"

    srt_lines = []
    for i, (s, e, txt) in enumerate(samples, 1):
        srt_lines.append(f"{i}\n{_fmt(s)} --> {_fmt(e)}\n{txt}\n")
    srt_clip = clip_dir / "clip_test.srt"
    srt_clip.write_text("\n".join(srt_lines), encoding="utf-8")

    ass_path  = clip_dir / "clip_test.ass"
    srt_to_ass(srt_clip, ass_path)

    final_mp4 = clip_dir / "clip_test.mp4"
    burn_captions(raw_mp4, ass_path, final_mp4)
    print(f"[TEST] ✓ {final_mp4}")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    args    = sys.argv[1:]
    preview = "--preview" in args
    args    = [a for a in args if a != "--preview"]

    only_clip = None
    if "--clip" in args:
        idx = args.index("--clip")
        only_clip = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    if not args:
        print("Usage:")
        print("  python clipper.py <video.mp4>                          # transcribe, then wait for MALIK")
        print("  python clipper.py <video.mp4> <malik.json>             # use existing MALIK output")
        print("  python clipper.py <video.mp4> <malik.json> --preview   # cut + vertical + captions for clip 1 only, then stop")
        print("  python clipper.py <video.mp4> --test                   # font/position test clip")
        print("  python clipper.py <video.mp4> <malik.json> --clip 3     # process only clip with priority 3")
        print("  python clipper.py reburn <clip_dir>                    # reburn after editing .ass")
        sys.exit(1)

    if len(args) == 2 and args[1] == "--test":
        run_test(Path(args[0]).expanduser().resolve())
        return

    if args[0] == "reburn":
        if len(args) < 2:
            print("Usage: python clipper.py reburn <clip_dir>")
            sys.exit(1)
        reburn(Path(args[1]))
        return

    video_path = Path(args[0]).expanduser().resolve()
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    out_dir = video_path.parent / f"{video_path.stem}_clips"
    out_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output folder: {out_dir}\n")

    # Step 1: Transcribe
    json_path = transcribe(video_path, out_dir)

    # Step 2: Load transcript entries for captions
    entries = parse_whisper_json(json_path)

    # Step 3: Get MALIK clip list
    if len(args) >= 2:
        malik_path = Path(args[1]).expanduser().resolve()
        if not malik_path.exists():
            print(f"[ERROR] MALIK JSON not found: {malik_path}")
            sys.exit(1)
    else:
        malik_path = out_dir / "malik_clips.json"
        if not malik_path.exists():
            print(f"\n[2/3] Transcript ready: {json_path}")
            print(f"\n  Run MALIK on the transcript above and save the output as:")
            print(f"  {malik_path}")
            print(f"\nPress Enter when malik_clips.json is in place...")
            input()
        if not malik_path.exists():
            print(f"[ERROR] {malik_path} not found. Run MALIK and save output there.")
            sys.exit(1)

    clips = json.loads(malik_path.read_text(encoding="utf-8"))["suggested_clips"]
    clips = sorted(clips, key=lambda x: x["priority"])

    if only_clip is not None:
        clips = [c for c in clips if c["priority"] == only_clip]
        if not clips:
            print(f"[ERROR] No clip with priority {only_clip} found.")
            sys.exit(1)
        print(f"[CLIP] Processing priority {only_clip} only...\n")
    elif preview:
        clips = clips[:1]
        print(f"[PREVIEW] Processing clip 1 of {len(clips)} only...\n")
    else:
        print(f"[3/3] Cutting {len(clips)} clips from MALIK output...\n")

    for clip in clips:
        start     = clip["start_sec"]
        end       = clip["end_sec"] if not preview else clip["start_sec"] + 20
        dur       = end - start
        clip_name = f"preview_p{clip['priority']:02d}" if preview else f"clip_p{clip['priority']:02d}"
        clip_dir  = out_dir / clip_name
        clip_dir.mkdir(exist_ok=True)

        print(f"  [p{clip['priority']}] {clip_name} | score {clip['score']} | "
              f"{start}s → {end}s ({dur:.0f}s)")
        print(f"         {clip['topic']}")

        raw_mp4      = cut_clip(video_path, start, end, clip_dir, clip_name)
        vertical_mp4 = make_vertical(raw_mp4, clip_dir, clip_name)

        clip_entries = [e for e in entries if e["end_sec"] > start and e["start_sec"] < end]
        srt_clip = clip_dir / f"{clip_name}.srt"
        srt_clip.write_text(entries_to_srt(clip_entries, start), encoding="utf-8")

        ass_path  = clip_dir / f"{clip_name}.ass"
        srt_to_ass(srt_clip, ass_path)

        final_mp4 = clip_dir / f"{clip_name}.mp4"
        burn_captions(vertical_mp4, ass_path, final_mp4)
        print(f"  ✓ {final_mp4.name}\n")

    if preview:
        print(f"[PREVIEW] Done. Check the clip before running the full batch:")
        print(f"  {final_mp4}")
        print(f"\nHappy with it? Run without --preview to process all clips.")
        return

    print(f"✅ Done. {len(clips)} clips in: {out_dir}")
    print("\nTo edit captions and reburn:")
    print("  1. Edit the .ass file inside any clip folder")
    print("  2. Run: python clipper.py reburn <clip_folder>")


if __name__ == "__main__":
    main()
