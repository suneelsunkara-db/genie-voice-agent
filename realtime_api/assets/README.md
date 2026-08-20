# Agent voice reference clip

`agent_voice_female.wav` and `agent_voice_male.wav` are the two selectable
reference clips. The choice is made once on Home, persisted in the browser, and
sent by **every** page at `session.start`, so greetings, fillers, tool responses,
and injected summaries all use the same selected speaker. `agent_voice.wav` is
the backward-compatible fallback while either variant is missing.

VoxCPM2 clones timbre from a reference clip and exposes no RNG seed, so this file
is the only thing that makes the voice reproducible. Without it each session
synthesizes its first turn unprompted, gets an arbitrary speaker, and clones that
for the rest of the call — so the agent sounds like a different person on every
call, and the Home greeting does not match the next page.

## Requirements

Enforced by `realtime_api/voice_identity.py`; a clip that fails any check is
ignored with a warning and the app falls back to per-session bootstrapping.

- WAV, mono, 16-bit PCM
- 2–20 seconds
- 48 kHz is preferred (VoxCPM2's native output rate)

## Building or replacing the clip

```bash
# 1. Render loudness-normalized candidates (each is a different random speaker).
python3 scripts/voice/generate_agent_voice.py generate --count 10 --target-dbfs -22

# 2. Hear a candidate speak several languages before committing to it.
python3 scripts/voice/generate_agent_voice.py audition \
    /tmp/agent_voice/candidate_03.wav --languages en-US,th-TH,id-ID,zh-CN

# 3. Install the two voices selected by ear.
python3 scripts/voice/generate_agent_voice.py install \
    /tmp/agent_voice/candidate_03.wav --variant female
python3 scripts/voice/generate_agent_voice.py install \
    /tmp/agent_voice/candidate_07.wav --variant male
```

VoxCPM2 does not accept a speaker-gender prompt and cannot turn one candidate
into male/female variants. Generate a pool and select one of each by ear.
Generation and installation normalize reference RMS to -22 dBFS with a -3 dBFS
peak ceiling, so selecting a timbre is no longer also selecting an arbitrary
volume. The `voice_id` is a hash of each file's bytes, so the two variants have
separate cache entries and cannot collide.

## Verifying it is live

Every `session.start` log line carries `voice_variant` and `voice_id`. A given
variant must always report its corresponding id on every page; a null means that
session is bootstrapping its own, which points at missing variant and fallback
clips.
