"""Tests for POST /api/reframe (one clip in, the same clip reframed to 9:16 out).

The render child process and ffprobe are stubbed: these tests pin the endpoint
contract (validation, response fields, fallback reporting, cleanup), not the
reframe engine itself.
"""

import asyncio
import os
import subprocess
import threading
import time

import httpx
import pytest

app_module = pytest.importorskip("app")
import layout_ranges  # noqa: E402

TRANSNET_NVENC_LOG = (
    "   🚀 Reframe engine v2 (ffmpeg-native render)\n"
    "   🎬 Scene engine: TransNetV2 — 2 scenes\n"
    "🎞️ [Encoder] video encoder: h264_nvenc (FFMPEG_ENCODER=auto)\n"
)
FALLBACK_LOG = (
    "   ⚠️ TransNetV2 scene detection failed (RuntimeError: boom) — falling back to PySceneDetect\n"
    "⚠️ [Encoder] FFMPEG_ENCODER=nvenc but h264_nvenc is not usable here — falling back to libx264\n"
    "🎞️ [Encoder] video encoder: libx264 (FFMPEG_ENCODER=nvenc)\n"
)


def _post(files=None):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/reframe", files=files)
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
        if path.endswith("_9x16.mp4"):
            return {"width": 1080, "height": 1920, "duration": 14.28, "has_audio": True}
        return {"width": 1920, "height": 1080, "duration": 14.28, "has_audio": True}
    monkeypatch.setattr(app_module, "_probe_video", probe)


def _fake_child(log_text, ranges=((0.0, 14.28, "TRACK"),), fail=None, sidecar=True, sleep=0):
    calls = []

    def child(input_path, output_path):
        calls.append((input_path, output_path))
        assert os.path.isfile(input_path)  # the upload is on disk while the render runs
        if sleep:
            time.sleep(sleep)
        if fail:
            raise RuntimeError(fail)
        with open(output_path, "wb") as f:
            f.write(b"video")
        if sidecar:
            layout_ranges.write(output_path, list(ranges))
        return log_text
    child.calls = calls
    return child


# --- the render's own log lines -> which engine / encoder ran ---------------

def test_signals_transnet_nvenc():
    assert app_module._reframe_signals(TRANSNET_NVENC_LOG) == {
        "scene_engine": "transnetv2", "video_encoder": "h264_nvenc"}


def test_signals_report_both_fallbacks():
    assert app_module._reframe_signals(FALLBACK_LOG) == {
        "scene_engine": "pyscenedetect", "video_encoder": "libx264"}


def test_signals_missing_lines_raise():
    with pytest.raises(RuntimeError, match="scene engine"):
        app_module._reframe_signals("🎞️ [Encoder] video encoder: h264_nvenc\n")
    with pytest.raises(RuntimeError, match="video encoder"):
        app_module._reframe_signals("   🎬 Scene engine: TransNetV2 — 1 scenes\n")


# --- endpoint ----------------------------------------------------------------

def test_reframe_ok(dirs, fake_probe, monkeypatch):
    child = _fake_child(TRANSNET_NVENC_LOG, ranges=[(0.0, 8.9, "TRACK"), (8.9, 14.28, "TRACK")])
    monkeypatch.setattr(app_module, "_run_reframe_child", child)
    r = _post(files={"file": ("clip 1.mp4", b"x" * 100, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == 1080 and body["height"] == 1920 and body["has_audio"] is True
    assert body["scene_engine"] == "transnetv2" and body["video_encoder"] == "h264_nvenc"
    assert body["layout_ranges"] == [{"start": 0.0, "end": 8.9, "layout": "track"},
                                     {"start": 8.9, "end": 14.28, "layout": "track"}]
    assert os.path.isfile(body["output_path"])
    folder = os.path.basename(os.path.dirname(body["output_path"]))
    assert folder.startswith("reframe_")
    assert body["video_url"] == f"/videos/{folder}/clip 1_9x16.mp4"
    # only the video is left: the upload and the layout sidecar are gone
    assert os.listdir(dirs["up"]) == []
    assert os.listdir(os.path.dirname(body["output_path"])) == ["clip 1_9x16.mp4"]


def test_reframe_reports_fallbacks(dirs, fake_probe, monkeypatch):
    monkeypatch.setattr(app_module, "_run_reframe_child", _fake_child(FALLBACK_LOG))
    r = _post(files={"file": ("c.mp4", b"x", "video/mp4")})
    assert r.status_code == 200, r.text
    assert r.json()["scene_engine"] == "pyscenedetect"
    assert r.json()["video_encoder"] == "libx264"


def test_render_failure_is_500_with_text(dirs, fake_probe, monkeypatch):
    monkeypatch.setattr(app_module, "_run_reframe_child",
                        _fake_child("", fail="reframe failed (exit 1):\nTraceback ... boom"))
    r = _post(files={"file": ("c.mp4", b"x", "video/mp4")})
    assert r.status_code == 500
    assert "boom" in r.json()["detail"]
    assert os.listdir(dirs["up"]) == []
    assert [d for d in os.listdir(dirs["out"]) if d.startswith("reframe_")] == []


def test_log_without_signals_is_500(dirs, fake_probe, monkeypatch):
    monkeypatch.setattr(app_module, "_run_reframe_child", _fake_child("no useful lines\n"))
    r = _post(files={"file": ("c.mp4", b"x", "video/mp4")})
    assert r.status_code == 500
    assert "scene engine" in r.json()["detail"]


def test_empty_upload_is_400(dirs, monkeypatch):
    child = _fake_child(TRANSNET_NVENC_LOG)
    monkeypatch.setattr(app_module, "_run_reframe_child", child)
    r = _post(files={"file": ("c.mp4", b"", "video/mp4")})
    assert r.status_code == 400
    assert child.calls == []
    assert os.listdir(dirs["up"]) == []


def test_unreadable_video_is_400(dirs, monkeypatch):
    def bad_probe(path):
        raise RuntimeError("moov atom not found")
    monkeypatch.setattr(app_module, "_probe_video", bad_probe)
    child = _fake_child(TRANSNET_NVENC_LOG)
    monkeypatch.setattr(app_module, "_run_reframe_child", child)
    r = _post(files={"file": ("c.mp4", b"junk", "video/mp4")})
    assert r.status_code == 400
    assert "moov atom" in r.json()["detail"]
    assert child.calls == []
    assert os.listdir(dirs["up"]) == []


def test_billing_mode_refuses(dirs, monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    r = _post(files={"file": ("c.mp4", b"x", "video/mp4")})
    assert r.status_code == 403


def test_missing_sidecar_is_500(dirs, fake_probe, monkeypatch):
    monkeypatch.setattr(app_module, "_run_reframe_child", _fake_child(TRANSNET_NVENC_LOG, sidecar=False))
    r = _post(files={"file": ("c.mp4", b"x", "video/mp4")})
    assert r.status_code == 500
    assert "no layout sidecar" in r.json()["detail"]
    assert [d for d in os.listdir(dirs["out"]) if d.startswith("reframe_")] == []


def test_timeout_reports_log_tail(monkeypatch):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired(a[0], 1800, output=b"first line\n   Analyzing Scenes: 50%\nlast line")
    monkeypatch.setattr(app_module.subprocess, "run", slow)
    with pytest.raises(RuntimeError, match="(?s)timed out after 1800s.*last line"):
        app_module._run_reframe_child("in.mp4", "out.mp4")


def test_reframes_queue_one_at_a_time(dirs, fake_probe, monkeypatch):
    monkeypatch.setattr(app_module, "reframe_semaphore", asyncio.Semaphore(1))
    busy = {"now": 0, "max": 0}
    lock = threading.Lock()
    inner = _fake_child(TRANSNET_NVENC_LOG)

    def child(input_path, output_path):
        with lock:
            busy["now"] += 1
            busy["max"] = max(busy["max"], busy["now"])
        time.sleep(0.3)
        try:
            return inner(input_path, output_path)
        finally:
            with lock:
                busy["now"] -= 1
    monkeypatch.setattr(app_module, "_run_reframe_child", child)

    async def both():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await asyncio.gather(*[
                client.post("/api/reframe", files={"file": (f"c{i}.mp4", b"x", "video/mp4")}) for i in range(3)])
    results = asyncio.run(both())
    assert [r.status_code for r in results] == [200, 200, 200]
    assert busy["max"] == 1
