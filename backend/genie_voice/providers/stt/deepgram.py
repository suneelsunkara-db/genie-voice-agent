"""Deepgram STT adapter.

Owns the Deepgram *streaming* contract:
  - normalize(): Deepgram live payload -> canonical TranscriptEvent
  - mock_events(): generate Deepgram-shaped streaming payloads (delegates to the
    mock generator) so the local pipeline exercises the REAL schema
  - stream(): live WebSocket transport (stub - wire up for deployment=live)

Reference shape (Deepgram streaming):
  { type, channel_index:[i,n], start, duration, is_final, speech_final,
    channel:{ alternatives:[{ transcript, confidence, words:[
      { word, start, end, confidence, speaker, punctuated_word } ] }] }, entities }
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator

from genie_voice.models.contracts import CanonicalWord, TranscriptEvent

from ..base import STTProvider


def _as_list(value: Any) -> list:
    """Coerce a value to a plain Python list.

    On Databricks the bronze row reaches `normalize` via mapInPandas, where Spark
    `array<...>` fields surface as numpy arrays. Truth-testing those (e.g.
    `x or []`, `if x`) raises "ambiguous truth value", so normalize never relies
    on a list-like's truthiness - it goes through here first. Plain dicts/lists
    (the offline path) pass through unchanged.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


class DeepgramSTT(STTProvider):
    name = "deepgram"

    def normalize(self, raw: dict[str, Any], *, call_id: str) -> TranscriptEvent:
        channel_obj = raw.get("channel", {}) or {}
        alts = _as_list(channel_obj.get("alternatives"))
        alt = alts[0] if alts else {}

        words: list[CanonicalWord] = []
        for w in _as_list(alt.get("words")):
            words.append(
                CanonicalWord(
                    text=w.get("punctuated_word", w.get("word", "")),
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    confidence=float(w.get("confidence", 0.0)),
                    speaker=w.get("speaker"),
                )
            )

        ci = _as_list(raw.get("channel_index"))
        channel_index = ci[0] if ci else 0
        start = float(raw.get("start", 0.0))
        duration = float(raw.get("duration", 0.0))
        speaker = words[0].speaker if words else channel_index

        return TranscriptEvent(
            call_id=call_id,
            channel=int(channel_index),
            start=start,
            end=start + duration,
            text=alt.get("transcript", ""),
            confidence=float(alt.get("confidence", 0.0)),
            is_final=bool(raw.get("is_final", False)),
            is_utterance_end=bool(raw.get("speech_final", False)),
            words=words,
            speaker=speaker,
            provider=self.name,
            raw=raw,
        )

    def mock_events(
        self, script: list[dict[str, Any]], render: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Render a vendor-neutral call script into Deepgram-shaped streaming
        payloads (interim updates -> final -> speech_final per turn).

        All Deepgram-specific shaping lives here, in the adapter.
        """
        render = render or {}
        channels = render.get("channels", {"agent": 0, "customer": 1})
        step = int(render.get("interim_words_step", 2))
        inject_low_conf = bool(render.get("inject_low_confidence", True))

        clock = 0.0
        for turn in script:
            channel = channels.get(turn.get("speaker", "customer"), 1)
            tokens = str(turn.get("text", "")).split()
            if not tokens:
                continue

            # Word-level timing (~0.28s/word) relative to call start.
            word_objs = []
            t = clock
            for i, tok in enumerate(tokens):
                start = t
                end = t + 0.28
                conf = 0.985
                if inject_low_conf and i % 7 == 3:
                    conf = 0.62  # realistic ASR confidence dip
                word_objs.append(
                    {
                        "word": tok.strip(".,?!").lower(),
                        "start": round(start, 4),
                        "end": round(end, 4),
                        "confidence": conf,
                        "speaker": channel,
                        "punctuated_word": tok,
                    }
                )
                t = end + 0.02

            turn_start = clock
            turn_end = t

            # Interim (is_final=false) events as the turn "grows".
            for k in range(step, len(word_objs), step):
                partial = word_objs[:k]
                yield self._event(
                    channel, turn_start, partial[-1]["end"] - turn_start,
                    partial, is_final=False, speech_final=False,
                )

            # Final segment for the turn.
            yield self._event(
                channel, turn_start, turn_end - turn_start,
                word_objs, is_final=True, speech_final=False,
            )
            # Utterance end (speech_final=true) marks the natural turn boundary.
            yield self._event(
                channel, turn_start, turn_end - turn_start,
                word_objs, is_final=True, speech_final=True,
            )
            clock = turn_end + 0.4  # gap before next turn

    def _event(self, channel, start, duration, words, *, is_final, speech_final):
        transcript = " ".join(w["punctuated_word"] for w in words)
        confidence = round(sum(w["confidence"] for w in words) / len(words), 4)
        return {
            "type": "Results",
            "channel_index": [channel, 2],
            "start": round(start, 4),
            "duration": round(duration, 4),
            "is_final": is_final,
            "speech_final": speech_final,
            "channel": {
                "alternatives": [
                    {"transcript": transcript, "confidence": confidence, "words": words}
                ]
            },
            "entities": [],
        }

    def stream(self, audio_chunks: Iterable[bytes]) -> Iterator[dict[str, Any]]:
        raise NotImplementedError(
            "Live Deepgram WebSocket streaming not yet wired. "
            "Set deployment=local, or implement the listen.live client here."
        )
