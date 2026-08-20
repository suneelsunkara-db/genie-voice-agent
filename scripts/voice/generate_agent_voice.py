"""Generate, audition and install the agent's ONE fixed voice.

VoxCPM2 invents a speaker whenever it synthesizes without a reference clip, and
it has no RNG seed to pin. So the app's voice is chosen the only way it can be:
generate a handful of unprompted candidates, listen to them, and commit the one
we want as the reference clip every session clones from.

Typical flow:

    # 1. Render candidates (each call comes out as a different speaker).
    python3 scripts/voice/generate_agent_voice.py generate --count 6

    # 2. Listen, then hear your favourite speak other languages.
    python3 scripts/voice/generate_agent_voice.py audition \\
        /tmp/agent_voice/candidate_03.wav --languages en-US,th-TH,id-ID,zh-CN

    # 3. Commit it as the app-wide voice.
    python3 scripts/voice/generate_agent_voice.py install /tmp/agent_voice/candidate_03.wav
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import wave
from array import array
from pathlib import Path

from realtime_api.config import RealtimeSettings
from realtime_api.serving_factory import build_serving
from realtime_api.voice_identity import load_voice_seed

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSET_PATH = REPO_ROOT / "realtime_api" / "assets" / "agent_voice.wav"
VARIANT_ASSET_PATHS = {
    "female": REPO_ROOT / "realtime_api" / "assets" / "agent_voice_female.wav",
    "male": REPO_ROOT / "realtime_api" / "assets" / "agent_voice_male.wav",
}

# The clip's own delivery becomes the agent's delivery on every later turn, so the
# seed line is written the way we want every answer to sound: calm, clear, unhurried.
SEED_LINE = (
    "Thank you for calling. I have your account details in front of me, "
    "and I'll walk you through exactly what happened, step by step."
)

# Rendered per language during an audition, so the same timbre can be judged
# across the languages the app actually supports.
AUDITION_INTENT = (
    "Reassure the customer that you have found their account and are reviewing "
    "the recent charges with them right now."
)


def _write_wav(path: Path, pcm: bytes, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm)


def _normalize_pcm(
    pcm: bytes, *, target_dbfs: float = -22.0, peak_ceiling_dbfs: float = -3.0
) -> bytes:
    """Give every seed the same restrained RMS level with a hard peak ceiling.

    Candidate 02 demonstrated why reference selection must not also be a volume
    lottery. Normalizing the *reference* before audition/install gives VoxCPM a
    consistent delivery level while preserving the candidate's pitch and timbre.
    """
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return pcm
    rms = math.sqrt(sum(float(s) * s for s in samples) / len(samples))
    peak = max(abs(s) for s in samples)
    if rms == 0 or peak == 0:
        return pcm
    full_scale = 32767.0
    target_rms = full_scale * (10 ** (target_dbfs / 20))
    peak_ceiling = full_scale * (10 ** (peak_ceiling_dbfs / 20))
    gain = min(target_rms / rms, peak_ceiling / peak)
    normalized = array("h", (max(-32768, min(32767, round(s * gain))) for s in samples))
    if sys.byteorder != "little":
        normalized.byteswap()
    return normalized.tobytes()


def _synthesize(serving, text: str, language: str, reference_b64: str | None) -> tuple[bytes, int]:
    chunks = list(
        serving.synthesize_stream(text, language=language, reference_audio_b64=reference_b64)
    )
    if not chunks:
        raise RuntimeError(f"TTS returned no audio for language={language}")
    return b"".join(c.pcm for c in chunks), chunks[0].sample_rate_hz


def _reference_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def cmd_generate(args: argparse.Namespace) -> None:
    serving = build_serving(RealtimeSettings.resolve())
    out = Path(args.out)
    manifest = []
    for i in range(1, args.count + 1):
        pcm, sample_rate_hz = _synthesize(serving, SEED_LINE, args.language, None)
        pcm = _normalize_pcm(pcm, target_dbfs=args.target_dbfs)
        path = out / f"candidate_{i:02d}.wav"
        _write_wav(path, pcm, sample_rate_hz)
        seconds = len(pcm) / 2 / sample_rate_hz
        # voice_id is reported now so the clip you pick can be matched against the
        # voice_id logged at session.start once it is installed.
        seed = load_voice_seed(str(path))
        voice_id = seed.voice_id if seed else None
        manifest.append(
            {"candidate": path.name, "seconds": round(seconds, 2),
             "sample_rate_hz": sample_rate_hz, "voice_id": voice_id}
        )
        print(f"{path}  {seconds:.2f}s  {sample_rate_hz}Hz  voice_id={voice_id}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{args.count} candidates in {out}. Listen, then audition your favourite.")


def cmd_audition(args: argparse.Namespace) -> None:
    clip = Path(args.clip)
    if not clip.is_file():
        raise SystemExit(f"no such clip: {clip}")
    settings = RealtimeSettings.resolve()
    serving = build_serving(settings)
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    reference_b64 = _reference_b64(clip)
    out = Path(args.out) / f"audition_{clip.stem}"
    for language in languages:
        text = serving.phrase(AUDITION_INTENT, language=language)
        pcm, sample_rate_hz = _synthesize(serving, text, language, reference_b64)
        path = out / f"{language}.wav"
        _write_wav(path, pcm, sample_rate_hz)
        print(f"{path}  {language}  {text}")
    print(f"\nAudition clips in {out}. Same timbre should carry across every language.")


def cmd_install(args: argparse.Namespace) -> None:
    clip = Path(args.clip)
    if not clip.is_file():
        raise SystemExit(f"no such clip: {clip}")
    asset_path = VARIANT_ASSET_PATHS.get(args.variant, ASSET_PATH)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(clip), "rb") as handle:
        sample_rate_hz = handle.getframerate()
        frame_count = (
            int(args.seconds * sample_rate_hz) if args.seconds else handle.getnframes()
        )
        pcm = handle.readframes(frame_count)
    pcm = _normalize_pcm(pcm, target_dbfs=args.target_dbfs)
    _write_wav(asset_path, pcm, sample_rate_hz)
    seed = load_voice_seed(str(asset_path))
    if seed is None:
        raise SystemExit(
            f"{asset_path} was written but is not a usable reference clip — see the "
            "warning above. The app would fall back to a per-session voice."
        )
    print(
        f"installed {asset_path.relative_to(REPO_ROOT)}"
        f"{f' ({args.variant})' if args.variant else ''}\n"
        f"  voice_id = {seed.voice_id}\n"
        f"  {seed.seconds:.2f}s  {seed.sample_rate_hz}Hz\n\n"
        "Every session now clones this clip. Confirm in the logs that session.start "
        "reports this same voice_id on every call."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="/tmp/agent_voice", help="working directory for clips")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="render N unprompted candidate voices")
    gen.add_argument("--count", type=int, default=6)
    gen.add_argument("--language", default="en-US", help="language of the seed line")
    gen.add_argument(
        "--target-dbfs",
        type=float,
        default=-22.0,
        help="RMS loudness for each reference candidate (default: -22 dBFS)",
    )
    gen.set_defaults(func=cmd_generate)

    aud = sub.add_parser("audition", help="hear one candidate speak several languages")
    aud.add_argument("clip")
    aud.add_argument("--languages", default="en-US,th-TH,id-ID,zh-CN")
    aud.set_defaults(func=cmd_audition)

    ins = sub.add_parser("install", help="commit a clip as the app-wide voice")
    ins.add_argument("clip")
    ins.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="trim to this many leading seconds (0 keeps the whole clip)",
    )
    ins.add_argument(
        "--variant",
        choices=sorted(VARIANT_ASSET_PATHS),
        help="install as one app-selectable variant; omit for the legacy fallback",
    )
    ins.add_argument(
        "--target-dbfs",
        type=float,
        default=-22.0,
        help="RMS loudness of the installed reference (default: -22 dBFS)",
    )
    ins.set_defaults(func=cmd_install)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
