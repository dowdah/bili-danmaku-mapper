import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bili_danmaku_mapper import AlignConfig, map_danmaku
from bili_danmaku_mapper.cli import main, parser, run_batch, run_pair
from bili_danmaku_mapper.io import atomic_json, save_mapping_outputs
from bili_danmaku_mapper.media import extract_features
from tests.test_mapping import alignment
from tests.video_helpers import make_video


def test_zero_single_multiple_xml():
    from bili_downloader import xml_2_json
    assert xml_2_json("<i/>") == ({"code": 1, "data": []}, 0)
    item = '<d p="1.5,4,25,255,0,0,uid,0">hello</d>'
    data, count = xml_2_json(f"<i>{item}</i>")
    assert count == 1
    assert data["data"][0]["type"] == "bottom"
    assert data["data"][0]["color"] == "#0000ff"
    assert xml_2_json(f"<i>{item}{item}</i>")[1] == 2


def test_legacy_keys_no_absorption_and_csv(tmp_path, monkeypatch):
    import mapper_frame
    monkeypatch.setattr(mapper_frame, "get_video_fps", lambda path: 1.0)
    data = {"data": [{"time": x} for x in range(4)]}
    original = copy.deepcopy(data)
    mapped = mapper_frame.map_danmaku_json(data, {"0": 0, "1": None, "2": 3, "3": None}, "a", "b")
    assert mapped["data"] == [{"time": 0}, {"time": 3}]
    assert data == original
    assert mapper_frame.map_danmaku_json(data, {}, "a", "b")["data"] == []
    file = tmp_path / "sparse.csv"
    mapper_frame.save_mapping_csv(file, {"10": 30, "20": None})
    assert list(csv.reader(file.open()))[1:] == [["10", "30"], ["20", "None"]]


def test_frame_numeric_sort_and_incomplete(tmp_path):
    from mapper_frame import load_frame_paths
    for name in ("99999.jpg", "100000.jpg", "100001.jpg"):
        (tmp_path / name).touch()
    assert [Path(p).stem for p in load_frame_paths(tmp_path)] == ["99999", "100000", "100001"]
    (tmp_path / "100000.jpg").unlink()
    with pytest.raises(ValueError):
        load_frame_paths(tmp_path)


def test_export_has_pts_and_refuses_stale_frames(tmp_path):
    from mapper_frame import export_frames_ffmpeg, videos_2_frames
    video = make_video(tmp_path / "0.mp4", 1)
    destination = tmp_path / "frames"
    export_frames_ffmpeg(str(video), str(destination))
    manifest = json.loads((destination / "frames.manifest.json").read_text())
    assert manifest["frame_count"] == 24
    assert len(manifest["times"]) == 24
    with pytest.raises(FileExistsError):
        export_frames_ffmpeg(str(video), str(destination))
    with pytest.raises(ValueError):
        videos_2_frames(str(tmp_path), delete_video=True)
    assert video.exists()


def test_atomic_outputs_preflight_and_failure(tmp_path):
    data = {"data": [{"time": 1.5}]}
    a = alignment([[(1, 1), (2, 2)]])
    result = map_danmaku(data, a)
    path = tmp_path / "out.json"
    save_mapping_outputs(path, result, a)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_mapping_outputs(path, result, a)
    result.report["bad"] = float("nan")
    with pytest.raises(ValueError):
        save_mapping_outputs(path, result, a, overwrite=True)
    assert path.read_bytes() == original
    assert not path.with_name(".out.json.lock").exists()


def test_output_commit_failure_rolls_back_all_files(tmp_path, monkeypatch):
    import bili_danmaku_mapper.io as output_io
    a = alignment([[(1, 1), (2, 2)]])
    result = map_danmaku({"data": [{"time": 1.5}]}, a)
    paths = save_mapping_outputs(tmp_path / "out.json", result, a)
    old = {path: path.read_bytes() for path in paths}
    actual = output_io.os.replace
    def fail_on_report(source, target):
        if Path(target) == paths[2]:
            raise OSError("Simulated report commit failure")
        return actual(source, target)
    monkeypatch.setattr(output_io.os, "replace", fail_on_report)
    result.danmaku["data"][0]["text"] = "changed"
    with pytest.raises(OSError):
        save_mapping_outputs(paths[0], result, a, overwrite=True)
    assert all(path.read_bytes() == old[path] for path in paths)


def test_legacy_entrypoint_forwards_supported_configuration(monkeypatch):
    import mapper_frame
    import bili_danmaku_mapper.cli as cli
    seen = []
    monkeypatch.setattr(cli, "run_pair", lambda *args: seen.append(args))
    mapper_frame.map_danmaku_main("target", "source", "raw", "mapped",
                                 f_config=mapper_frame.FrameExportConfig(crop_ratio=.9),
                                 aap_config=mapper_frame.AAPConfig(max_hamming_distance=4))
    config = seen[0][4]
    assert config.crop_ratio == .9 and config.max_hash_distance == 4


def test_parallel_atomic_writers(tmp_path):
    path = tmp_path / "out.json"
    def write(index):
        try:
            atomic_json(path, {"writer": index})
            return True
        except FileExistsError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(write, range(4))) == 1
    assert json.loads(path.read_text())["writer"] in range(4)


def test_parallel_features_cache(tmp_path):
    video = make_video(tmp_path / "video.mp4", 2)
    config = AlignConfig(cache_dir=str(tmp_path / "cache"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda _: extract_features(video, config), range(2)))
    assert np.array_equal(values[0].hashes, values[1].hashes)
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1


def test_cli_batch_sparse_episodes_and_pairs(tmp_path):
    source, target, output = [tmp_path / name for name in ("source", "target", "output")]
    source.mkdir(); target.mkdir()
    first = make_video(source / "1.mp4", 2)
    (source / "3.mp4").write_bytes(first.read_bytes())
    (target / "2.mp4").write_bytes(first.read_bytes())
    (target / "4.mp4").write_bytes(first.read_bytes())
    for no in (1, 3):
        atomic_json(source / f"dms_raw_{no}.json", {"code": 1, "data": [{"time": .5, "text": "hi"}]})
    pairs = tmp_path / "pairs.json"
    atomic_json(pairs, [{"source": 1, "target": 2}, {"source": 3, "target": 4}, {"source": 7, "target": 8}])
    args = parser().parse_args(["batch", "--source-dir", str(source), "--target-dir", str(target),
                               "--output-dir", str(output), "--pairs", str(pairs)])
    report = run_batch(args, AlignConfig(cache_dir=str(tmp_path / "cache")))
    assert report["counts"] == {"success": 2, "failed": 0, "skipped": 1}
    assert (output / "dms_ep_4.report.json").exists()
    assert not list(output.glob("*.lock"))


def test_cli_identity_and_input_protection(tmp_path):
    video = make_video(tmp_path / "video.mp4", 2)
    data = tmp_path / "raw.json"
    atomic_json(data, {"data": [{"time": .7}]})
    args = ["align", "--source", str(video), "--target", str(video), "--danmaku", str(data),
            "--cache-dir", str(tmp_path / "cache"), "--output", str(tmp_path / "out.json")]
    assert main(args) == 0
    assert main(args) == 1
    assert main(args + ["--overwrite"]) == 0
    with pytest.raises(ValueError):
        run_pair(video, video, data, data, AlignConfig(), "strict", True)


def test_config_precedence(tmp_path, monkeypatch):
    file = tmp_path / ".env"
    file.write_text("SAMPLE_RATE=3\nCROP_RATIO=.9\nBASE_DIR=/tmp/features-test\n")
    monkeypatch.setenv("SAMPLE_RATE", "4")
    assert AlignConfig.resolve(file).sample_rate == 4
    config = AlignConfig.resolve(file, sample_rate=5)
    assert config.sample_rate == 5 and config.crop_ratio == .9
    assert config.cache_dir == "/tmp/features-test/features"


def test_failed_download_does_not_publish_partial(tmp_path, monkeypatch):
    import bili_downloader
    class Client:
        async def download_create(self, *args):
            return 0
        def download_content_length(self, _):
            return 4
        async def download_chunk(self, _):
            return b""
    monkeypatch.setattr(bili_downloader, "get_client", lambda: Client())
    output = tmp_path / "download.bin"
    with pytest.raises(RuntimeError):
        asyncio.run(bili_downloader.download("signed-url-not-logged", str(output), "test", retries=1))
    assert not output.exists()
    assert not list(tmp_path.glob(".download-*"))


@pytest.mark.parametrize("known_length", [True, False])
def test_complete_download_handles_iterator_eof(tmp_path, monkeypatch, known_length):
    import bili_downloader
    class Client:
        def __init__(self):
            self.chunks = iter([b"ab", b"cd"])
        async def download_create(self, *args):
            return 0
        def download_content_length(self, _):
            return 4 if known_length else 0
        async def download_chunk(self, _):
            try:
                return next(self.chunks)
            except StopIteration:
                raise StopAsyncIteration
    client = Client()
    monkeypatch.setattr(bili_downloader, "get_client", lambda: client)
    output = tmp_path / "complete.bin"
    asyncio.run(bili_downloader.download("fake-url", str(output), "test", retries=1))
    assert output.read_bytes() == b"abcd"


def test_refreshed_credentials_saved_privately(tmp_path, monkeypatch):
    import bili_downloader
    cookies = tmp_path / "cookies.json"
    atomic_json(cookies, {"SESSDATA": "old-test-value"})
    class FakeCredential:
        def __init__(self, **kwargs):
            self.value = "old-test-value"
        async def check_valid(self):
            return True
        async def check_refresh(self):
            return True
        async def refresh(self):
            self.value = "new-test-value"
        def get_cookies(self):
            return {"SESSDATA": self.value}
    async def noop(*args):
        pass
    monkeypatch.setattr(bili_downloader, "Credential", FakeCredential)
    monkeypatch.setattr(bili_downloader, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(bili_downloader, "cookies_path", str(cookies))
    monkeypatch.setattr(bili_downloader, "download_bangumi", noop)
    asyncio.run(bili_downloader._run(1, False))
    assert json.loads(cookies.read_text())["SESSDATA"] == "new-test-value"
    assert cookies.stat().st_mode & 0o777 == 0o600
