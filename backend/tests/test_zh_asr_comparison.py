from types import SimpleNamespace
from unittest.mock import patch

from genie_voice.asr.zh_comparison import list_recent_comparisons, schedule_zh_asr_comparison


def test_schedule_zh_asr_comparison_stores_running_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = SimpleNamespace(
        providers=SimpleNamespace(
            stt=SimpleNamespace(
                active_options=lambda: {
                    "routes": {
                        "zh-CN": {"endpoint": "voice_asr_zh_oss_qwen3_asr_0_6b"},
                        "zh-CN-sensevoice": {"endpoint": "voice_asr_zh_oss_sensevoice_small"},
                        "zh-CN-paraformer": {"endpoint": "voice_asr_zh_oss_paraformer_8k"},
                    }
                }
            )
        )
    )

    with patch("genie_voice.asr.zh_comparison._query_endpoint") as query:
        query.side_effect = [
            ("sensevoice text", None),
            ("paraformer text", None),
            ("qwen text", None),
        ]
        comparison_id = schedule_zh_asr_comparison(
            call_id="CALL-2028",
            audio_b64="ZmFrZQ==",
            mime_type="audio/webm",
            speaker=1,
            selected_language="zh-CN-sensevoice",
            primary_transcript="sensevoice text",
            settings=settings,
            browser_caption="浏览器字幕",
        )

    assert comparison_id
    import time

    deadline = time.time() + 2.0
    items = []
    while time.time() < deadline:
        items = list_recent_comparisons(call_id="CALL-2028")
        if items and items[0].get("status") == "done":
            break
        time.sleep(0.05)

    assert items
    latest = items[0]
    assert latest["comparison_id"] == comparison_id
    assert latest["browser_caption"] == "浏览器字幕"
    assert latest["primary_transcript"] == "sensevoice text"
    assert len(latest["models"]) == 3
    assert query.call_count == 3
