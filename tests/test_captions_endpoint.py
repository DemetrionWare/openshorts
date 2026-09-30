"""Tests for POST /api/captions (one video + its caption words in, the captioned video out).

The burn child process and ffprobe are stubbed: these tests pin the endpoint
contract (validation, response fields, what the render's report proves,
cleanup), not ffmpeg or libass themselves. The ASS file is real: it comes from
subtitles.generate_ass, and the tests read what the child was handed.
"""

import asyncio
import json
import os
import subprocess
import threading
import time

import httpx
import pytest

app_module = pytest.importorskip("app")

NVENC_REPORT = (
    "  Stream #0:0 -> #0:0 (h264 (native) -> h264 (h264_nvenc))\n"
    "  Stream #0:1 -> #0:1 (copy)\n"
    "frame=  97 fps=0.0 q=11.0 [Parsed_ass_0 @ 0000025e] fontselect: (Anton, 700, 0) -> Anton-Regular, 0, Anton-Regular\n"
)
X264_REPORT = NVENC_REPORT.replace("h264 (h264_nvenc)", "h264 (libx264)")
WRONG_FONT_REPORT = NVENC_REPORT.replace("-> Anton-Regular, 0, Anton-Regular", "-> Arial-BoldMT, 0, Arial-BoldMT")
GLYPH_REPORT = NVENC_REPORT + (
    "[Parsed_ass_0 @ 0000025e] Glyph 0x2603 not found, selecting one more font for (Anton, 700, 0)\n"
    "[Parsed_ass_0 @ 0000025e] fontselect: (Anton, 700, 0) -> SegoeUIEmoji, 0, SegoeUIEmoji\n"
)

# The look the user picked (AUTO_CAPTION_STYLE), at margin_v 64.
LOOK = {
    "position": "bottom", "margin_v": "64", "font_name": "Anton", "font_size": "44",
    "font_color": "#FFFFFF", "border_color": "#000000", "border_width": "4",
    "style": "karaoke", "highlight_color": "#FFE500", "effect": "pop",
    "base_opacity": "1.0", "uppercase": "true", "bg_color": "#000000", "bg_opacity": "0.0",
    "max_chars": "16", "max_duration": "1.4",
}
WORDS = [{"text": "He's", "start": 0.5, "end": 0.8}, {"text": "RIGHT,", "start": 0.8, "end": 1.2},
         {"text": "that", "start": 1.3, "end": 1.5}, {"text": "really", "start": 1.5, "end": 1.9},
         {"text": "happened.", "start": 1.9, "end": 2.6}]


def _form(words=None, **changes):
    data = dict(LOOK, words=json.dumps(WORDS if words is None else words))
    for k, v in changes.items():
        if v is None:
            data.pop(k)
        else:
            data[k] = v
    return data


def _post(data, files=None):
    if files is None:
        files = {"file": ("01_Scanning a Book Doesn't.mp4", b"x" * 100, "video/mp4")}

    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/captions", data=data, files=files)
    return asyncio.run(_do())


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    out_root = tmp_path / "output"
    up_root = tmp_path / "uploads"
    out_root.mkdir()
    up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))
    return {"out": out_root, "up": up_root}


@pytest.fixture()
def fake_probe(monkeypatch):
    def probe(path):
        return {"width": 1080, "height": 1920, "duration": 66.316, "has_audio": True}
    monkeypatch.setattr(app_module, "_probe_video", probe)


def _fake_child(report=NVENC_REPORT, fail=None, write_output=True, sleep=0):
    calls = []

    def child(input_path, ass_path, output_path, report_path):
        with open(ass_path, encoding="utf-8-sig") as f:
            calls.append({"input": input_path, "ass": f.read(), "output": output_path})
        assert os.path.isfile(input_path)  # the upload is on disk while the burn runs
        if sleep:
            time.sleep(sleep)
        if fail:
            raise RuntimeError(fail)
        if write_output:
            with open(output_path, "wb") as f:
                f.write(b"video")
        if report is not None:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
        return "🎞️ [Encoder] video encoder: h264_nvenc (FFMPEG_ENCODER=auto)\n"
    child.calls = calls
    return child


def _leftovers(dirs):
    return {"uploads": os.listdir(dirs["up"]),
            "captions": [d for d in os.listdir(dirs["out"]) if d.startswith("captions_")]}


# --- words ---------------------------------------------------------------------

def test_words_ok():
    assert app_module._parse_caption_words(json.dumps(WORDS)) == [
        {"text": w["text"], "start": float(w["start"]), "end": float(w["end"])} for w in WORDS]


def test_words_refused():
    cases = [
        ("not json", "not valid JSON"),
        ("[]", "non-empty JSON array"),
        ('{"text": "a"}', "non-empty JSON array"),
        ('[{"text": "a", "start": 0}]', "word 0 must be an object with exactly the keys"),
        ('[{"text": "a", "start": 0, "end": 1, "startMs": 0}]', "exactly the keys"),
        ('[{"text": "", "start": 0, "end": 1}]', "non-empty string"),
        ('[{"text": " a", "start": 0, "end": 1}]', "leading, trailing or repeated whitespace"),
        ('[{"text": "a  b", "start": 0, "end": 1}]', "repeated whitespace"),
        ('[{"text": "a{b", "start": 0, "end": 1}]', "ASS control characters"),
        ('[{"text": "a\\\\b", "start": 0, "end": 1}]', "ASS control characters"),
        ('[{"text": 5, "start": 0, "end": 1}]', "non-empty string"),
        ('[{"text": "a", "start": "0", "end": 1}]', "start must be a number"),
        ('[{"text": "a", "start": true, "end": 1}]', "start must be a number"),
        ('[{"text": "a", "start": 0, "end": NaN}]', "end must be a number"),
        ('[{"text": "a", "start": 0, "end": Infinity}]', "end must be a number"),
        ('[{"text": "a", "start": -0.1, "end": 1}]', "0 <= start < end"),
        ('[{"text": "a", "start": 1, "end": 1}]', "0 <= start < end"),
        ('[{"text": "a", "start": 1, "end": 2}, {"text": "b", "start": 1, "end": 3}]',
         "word 1 \\('b'\\) starts at 1, not after word 0's start 1"),
        ('[{"text": "a", "start": 1, "end": 2}, {"text": "b", "start": 0.5, "end": 3}]', "not after word 0"),
    ]
    for raw, match in cases:
        with pytest.raises(ValueError, match=match):
            app_module._parse_caption_words(raw)


# --- style ---------------------------------------------------------------------

def _style(**changes):
    s = dict(position="bottom", margin_v=64, font_name="Anton", font_size=44, font_color="#FFFFFF",
             border_color="#000000", border_width=4, style="karaoke", highlight_color="#FFE500",
             effect="pop", base_opacity=1.0, bg_color="#000000", bg_opacity=0.0, max_chars=16,
             max_duration=1.4)
    s.update(changes)
    return s


def test_style_ok():
    app_module._check_caption_style(**_style())
    app_module._check_caption_style(**_style(font_size=12, margin_v=0, border_width=1,
                                             base_opacity=0.05, bg_opacity=1.0, max_chars=1,
                                             font_name="Noto Serif", position="top", effect="none"))


def test_style_refused():
    cases = [
        (dict(style="classic"), "style must be 'karaoke'"),
        (dict(position="Bottom"), "position must be one of top, middle, bottom"),
        (dict(effect="bounce"), "effect must be one of"),
        (dict(font_name="Anton,Bold"), "font_name must be"),
        (dict(font_name=" Anton"), "font_name must be"),
        (dict(font_name="Noto  Serif"), "font_name must be"),
        (dict(font_color="FFFFFF"), "font_color must be #RRGGBB"),
        (dict(border_color="#00000"), "border_color must be #RRGGBB"),
        (dict(highlight_color="#FFE50G"), "highlight_color must be #RRGGBB"),
        (dict(bg_color="black"), "bg_color must be #RRGGBB"),
        (dict(margin_v=201), "margin_v must be between 0 and 200"),
        (dict(margin_v=-1), "margin_v must be between 0 and 200"),
        (dict(font_size=11), "font_size must be between 12 and 200"),
        (dict(font_size=201), "font_size must be between 12 and 200"),
        (dict(border_width=0), "border_width must be between 1 and 10"),
        (dict(base_opacity=0.0), "base_opacity must be between 0.05 and 1.0"),
        (dict(base_opacity=float("nan")), "base_opacity must be between"),
        (dict(bg_opacity=1.5), "bg_opacity must be between 0.0 and 1.0"),
        (dict(max_chars=0), "max_chars must be at least 1"),
        (dict(max_duration=0.0), "max_duration must be"),
        (dict(max_duration=float("inf")), "max_duration must be"),
    ]
    for changes, match in cases:
        with pytest.raises(ValueError, match=match):
            app_module._check_caption_style(**_style(**changes))


def test_style_lists_every_problem():
    with pytest.raises(ValueError, match="style must be.*; position must be.*; font_color must be"):
        app_module._check_caption_style(**_style(style="classic", position="side", font_color="red"))


# --- the ASS generate_ass writes ------------------------------------------------

def _ass_for(tmp_path, words):
    path = str(tmp_path / "t.ass")
    transcript = {"segments": [{"words": [
        {"word": " " + w["text"], "start": w["start"], "end": w["end"]} for w in words]}]}
    assert app_module.generate_ass(transcript, 0.0, 66.0, path, max_chars=16, max_duration=1.4)
    return path


def test_events_ok(tmp_path):
    words = app_module._parse_caption_words(json.dumps(WORDS))
    app_module._check_caption_events(_ass_for(tmp_path, words), words)


def test_events_zero_length_after_rounding(tmp_path):
    words = [{"text": "a", "start": 1.001, "end": 1.5}, {"text": "b", "start": 1.004, "end": 1.6}]
    with pytest.raises(ValueError, match="word 0 \\('a'\\) would show for 0 s"):
        app_module._check_caption_events(_ass_for(tmp_path, words), words)
    words = [{"text": "a", "start": 1.0, "end": 1.004}]
    with pytest.raises(ValueError, match="word 0"):
        app_module._check_caption_events(_ass_for(tmp_path, words), words)


def test_events_count_mismatch(tmp_path):
    words = app_module._parse_caption_words(json.dumps(WORDS))
    with pytest.raises(RuntimeError, match="5 caption events for 6 words"):
        app_module._check_caption_events(_ass_for(tmp_path, words), words + [words[-1]])


# --- the render's report -> encoder and font ------------------------------------

def test_signals_nvenc_and_font():
    assert app_module._captions_signals(NVENC_REPORT, "Anton") == {
        "video_encoder": "h264_nvenc", "font": "Anton-Regular"}


def test_signals_report_libx264():
    assert app_module._captions_signals(X264_REPORT, "Anton")["video_encoder"] == "libx264"


def test_signals_font_substituted_is_value_error():
    with pytest.raises(ValueError, match="'Antonn' was not used as asked: libass rendered with Arial-BoldMT"):
        app_module._captions_signals(WRONG_FONT_REPORT.replace("(Anton,", "(Antonn,"), "Antonn")


def test_signals_missing_glyph_is_value_error():
    with pytest.raises(ValueError, match="(?s)rendered with SegoeUIEmoji.*no glyph for U\\+2603"):
        app_module._captions_signals(GLYPH_REPORT, "Anton")


def test_signals_missing_lines_raise():
    with pytest.raises(RuntimeError, match="which video encoder"):
        app_module._captions_signals(NVENC_REPORT.replace("(h264_nvenc)", ""), "Anton")
    with pytest.raises(RuntimeError, match="which font"):
        app_module._captions_signals("  Stream #0:0 -> #0:0 (h264 (native) -> h264 (h264_nvenc))\n", "Anton")


def test_font_family_match():
    ok = [("Anton", "Anton-Regular"), ("Noto Serif", "NotoSerif-Bold"),
          ("Liberation Sans", "LiberationSans-Bold"), ("Arial", "Arial-BoldMT")]
    bad = [("Antonn", "Arial-BoldMT"), ("Impact", "Anton-Regular"), ("Anton", "SegoeUIEmoji")]
    assert all(app_module._font_is_family(f, p) for f, p in ok)
    assert not any(app_module._font_is_family(f, p) for f, p in bad)


# --- endpoint ------------------------------------------------------------------

def test_captions_ok(dirs, fake_probe, monkeypatch):
    child = _fake_child()
    monkeypatch.setattr(app_module, "_run_captions_child", child)
    r = _post(_form())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == 1080 and body["height"] == 1920 and body["has_audio"] is True
    assert body["source"]["duration"] == 66.316
    assert body["video_encoder"] == "h264_nvenc" and body["font"] == "Anton-Regular"
    assert body["word_count"] == 5
    folder = os.path.basename(os.path.dirname(body["output_path"]))
    assert folder.startswith("captions_")
    assert body["video_url"] == f"/videos/{folder}/01_Scanning%20a%20Book%20Doesn%27t_captions.mp4"
    # only the video is left: the upload, the ASS and the ffmpeg report are gone
    assert os.listdir(dirs["up"]) == []
    assert os.listdir(os.path.dirname(body["output_path"])) == ["01_Scanning a Book Doesn't_captions.mp4"]
    # the ASS carries the requested look: Anton, ASS size int(44 * 0.85) = 37, white
    # fill, black outline 4, bottom (2), MarginV 64; words uppercased, spoken word yellow + pop
    ass = child.calls[0]["ass"]
    assert "Style: Default,Anton,37,&H00FFFFFF,&H00FFFFFF,&H00000000,&HFF000000,1,0,0,0,100,100,0,0,1,4,0,2,10,10,64,1" in ass
    assert "{\\c&H00E5FF&\\fscx90\\fscy90\\t(0,110,\\fscx108\\fscy108)}HE'S{\\r} RIGHT," in ass
    assert ass.count("Dialogue:") == 5


def test_captions_reports_libx264(dirs, fake_probe, monkeypatch):
    monkeypatch.setattr(app_module, "_run_captions_child", _fake_child(report=X264_REPORT))
    r = _post(_form())
    assert r.status_code == 200, r.text
    assert r.json()["video_encoder"] == "libx264"


def test_missing_field_is_422(dirs, fake_probe, monkeypatch):
    child = _fake_child()
    monkeypatch.setattr(app_module, "_run_captions_child", child)
    r = _post(_form(margin_v=None))
    assert r.status_code == 422
    assert "margin_v" in r.text
    r = _post(_form(font_size="44.5"))
    assert r.status_code == 422
    assert child.calls == []
    assert _leftovers(dirs) == {"uploads": [], "captions": []}


def test_bad_words_and_style_are_400(dirs, fake_probe, monkeypatch):
    child = _fake_child()
    monkeypatch.setattr(app_module, "_run_captions_child", child)
    r = _post(_form(words=[{"text": "a", "start": 2, "end": 1}]))
    assert r.status_code == 400 and "0 <= start < end" in r.json()["detail"]
    r = _post(_form(effect="bounce", bg_opacity="2"))
    assert r.status_code == 400
    assert "effect must be one of" in r.json()["detail"] and "bg_opacity" in r.json()["detail"]
    assert child.calls == []
    assert _leftovers(dirs) == {"uploads": [], "captions": []}


def test_word_after_video_end_is_400(dirs, fake_probe, monkeypatch):
    child = _fake_child()
    monkeypatch.setattr(app_module, "_run_captions_child", child)
    words = WORDS + [{"text": "late", "start": 66.316, "end": 67.0}]
    r = _post(_form(words=words))
    assert r.status_code == 400
    assert "word 5 ('late') at 66.316" in r.json()["detail"]
    assert child.calls == []
    assert _leftovers(dirs) == {"uploads": [], "captions": []}


def test_word_ending_after_video_end_is_ok(dirs, fake_probe, monkeypatch):
    monkeypatch.setattr(app_module, "_run_captions_child", _fake_child())
    r = _post(_form(words=WORDS + [{"text": "end", "start": 66.0, "end": 66.5}]))
    assert r.status_code == 200, r.text


def test_words_in_same_hundredth_is_400(dirs, fake_probe, monkeypatch):
    child = _fake_child()
    monkeypatch.setattr(app_module, "_run_captions_child", child)
    r = _post(_form(words=[{"text": "a", "start": 1.001, "end": 1.5}, {"text": "b", "start": 1.004, "end": 1.6}]))
    assert r.status_code == 400 and "1/100 s" in r.json()["detail"]
    assert child.calls == []
    assert _leftovers(dirs) == {"uploads": [], "captions": []}


def test_empty_upload_is_400(dirs, monkeypatch):
    child = _fake_child()
    monkeypatch.setattr(app_module, "_run_captions_child", child)
    r = _post(_form(), files={"file": ("c.mp4", b"", "video/mp4")})
    assert r.status_code == 400 and "empty" in r.json()["detail"]
    assert child.calls == []
    assert _leftovers(dirs) == {"uploads": [], "captions": []}


def test_unreadable_video_is_400(dirs, monkeypatch):
    def bad_probe(path):
        raise RuntimeError("moov atom not found")
    monkeypatch.setattr(app_module, "_probe_video", bad_probe)
    child = _fake_child()
    monkeypatch.setattr(app_module, "_run_captions_child", child)
    r = _post(_form(), files={"file": ("c.mp4", b"junk", "video/mp4")})
    assert r.status_code == 400 and "moov atom" in r.json()["detail"]
    assert child.calls == []
    assert _leftovers(dirs) == {"uploads": [], "captions": []}


def test_billing_mode_refuses(dirs, monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    r = _post(_form())
    assert r.status_code == 403


def test_burn_failure_is_500_with_log_tail(dirs, fake_probe, monkeypatch):
    monkeypatch.setattr(app_module, "_run_captions_child",
                        _fake_child(fail="caption burn failed (exit 1):\nException: FFmpeg failed: boom"))
    r = _post(_form())
    assert r.status_code == 500 and "boom" in r.json()["detail"]
    assert _leftovers(dirs) == {"uploads": [], "captions": []}


def test_font_substitution_is_400_and_deletes_output(dirs, fake_probe, monkeypatch):
    monkeypatch.setattr(app_module, "_run_captions_child", _fake_child(report=WRONG_FONT_REPORT))
    r = _post(_form())
    assert r.status_code == 400 and "Arial-BoldMT" in r.json()["detail"]
    assert _leftovers(dirs) == {"uploads": [], "captions": []}


def test_missing_report_or_output_is_500(dirs, fake_probe, monkeypatch):
    monkeypatch.setattr(app_module, "_run_captions_child", _fake_child(report=None))
    r = _post(_form())
    assert r.status_code == 500 and "ffmpeg_report.log" in r.json()["detail"]
    monkeypatch.setattr(app_module, "_run_captions_child", _fake_child(write_output=False))
    r = _post(_form())
    assert r.status_code == 500 and "wrote no file" in r.json()["detail"]
    assert _leftovers(dirs) == {"uploads": [], "captions": []}


def test_timeout_reports_log_tail(monkeypatch):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired(a[0], 1800, output=b"first line\nlast line")
    monkeypatch.setattr(app_module.subprocess, "run", slow)
    with pytest.raises(RuntimeError, match="(?s)timed out after 1800s.*last line"):
        app_module._run_captions_child("in.mp4", "c.ass", "out.mp4", "r.log")


def test_child_gets_escaped_report_path(monkeypatch):
    seen = {}

    class Done:
        returncode = 0
        stdout = "ok"

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return Done()
    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    app_module._run_captions_child("C:\\v\\in.mp4", "C:\\o\\c.ass", "C:\\o\\out.mp4", "C:\\o\\ffmpeg_report.log")
    assert seen["cmd"][-4:] == ["C:\\v\\in.mp4", "C:\\o\\c.ass", "C:\\o\\out.mp4",
                                "file=C\\:/o/ffmpeg_report.log:level=32"]


def test_captions_queue_one_at_a_time(dirs, fake_probe, monkeypatch):
    monkeypatch.setattr(app_module, "captions_semaphore", asyncio.Semaphore(1))
    busy = {"now": 0, "max": 0}
    lock = threading.Lock()
    inner = _fake_child()

    def child(*args):
        with lock:
            busy["now"] += 1
            busy["max"] = max(busy["max"], busy["now"])
        time.sleep(0.3)
        try:
            return inner(*args)
        finally:
            with lock:
                busy["now"] -= 1
    monkeypatch.setattr(app_module, "_run_captions_child", child)

    async def three():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await asyncio.gather(*[
                client.post("/api/captions", data=_form(),
                            files={"file": (f"c{i}.mp4", b"x", "video/mp4")}) for i in range(3)])
    results = asyncio.run(three())
    assert [r.status_code for r in results] == [200, 200, 200]
    assert busy["max"] == 1
