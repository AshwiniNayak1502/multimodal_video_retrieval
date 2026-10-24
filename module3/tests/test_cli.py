"""Integration tests for Module 3 CLI commands using synthetic test environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure module3 directory is in sys.path
_MODULE3_DIR = Path(__file__).resolve().parent.parent
if str(_MODULE3_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE3_DIR))

from main import main


def test_cli_health_command(
    synthetic_cli_config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify 'health --config ...' command runs against synthetic index and reports readiness."""
    test_args = ["main.py", "health", "--config", str(synthetic_cli_config_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "HEALTH CHECK" in captured.out
    assert "READY" in captured.out


def test_cli_health_command_json(
    synthetic_cli_config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify 'health --json --config ...' outputs valid JSON with schema version 3.1.0."""
    test_args = ["main.py", "health", "--config", str(synthetic_cli_config_file), "--json"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "ready"
    assert data["indexed_videos"] >= 2
    assert data["schema_version"] == "3.1.0"


def test_cli_videos_command(
    synthetic_cli_config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify 'videos --config ...' command lists indexed videos from synthetic index."""
    test_args = ["main.py", "videos", "--config", str(synthetic_cli_config_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "INDEXED VIDEOS" in captured.out
    assert "vid_test001" in captured.out
    assert "vid_silent002" in captured.out


def test_cli_search_command_json(
    synthetic_cli_config_file: Path,
    indexed_test_environment: Tuple[Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify 'search --config ... --json' outputs valid SearchResponse JSON."""
    service, _, _, _ = indexed_test_environment
    monkeypatch.setattr("main.QueryService", lambda config=None, video_index=None, warmup=False: service)

    test_args = [
        "main.py",
        "search",
        "multimodal indexing",
        "--config",
        str(synthetic_cli_config_file),
        "--top-k",
        "5",
        "--json",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] in ("ok", "no_match")
    assert "results" in data
    assert "timings_ms" in data
    assert "notices" in data
    assert data["schema_version"] == "3.1.0"


def test_cli_search_command_debug(
    synthetic_cli_config_file: Path,
    indexed_test_environment: Tuple[Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify 'search --config ... --debug' outputs stage latencies and diagnostics."""
    service, _, _, _ = indexed_test_environment
    monkeypatch.setattr("main.QueryService", lambda config=None, video_index=None, warmup=False: service)

    test_args = [
        "main.py",
        "search",
        "multimodal indexing",
        "--config",
        str(synthetic_cli_config_file),
        "--debug",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "Execution Latency" in captured.out
    assert "Debug Diagnostics" in captured.out


def test_cli_config_missing_file_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify non-existent --config path exits cleanly with friendly error."""
    test_args = ["main.py", "health", "--config", "non_existent_config.json"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error: Specified configuration file not found" in captured.err


def test_cli_config_malformed_json_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify malformed JSON in --config file exits cleanly with friendly error."""
    bad_cfg = tmp_path / "bad_syntax.json"
    bad_cfg.write_text("{ this is not valid json", encoding="utf-8")

    test_args = ["main.py", "health", "--config", str(bad_cfg)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Failed to load configuration" in captured.err


def test_cli_config_unknown_key_error(
    synthetic_cli_config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify unknown keys in config file are rejected with friendly error."""
    with open(synthetic_cli_config_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["unknown_forbidden_option"] = 12345
    unknown_cfg = tmp_path / "unknown_key.json"
    with open(unknown_cfg, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    test_args = ["main.py", "health", "--config", str(unknown_cfg)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Unknown configuration" in captured.err


def test_cli_search_top_k_flag_overrides_and_bounds(
    synthetic_cli_config_file: Path,
    indexed_test_environment: Tuple[Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI --top-k bounds (3-5 accepted, others rejected)."""
    service, _, _, _ = indexed_test_environment
    monkeypatch.setattr("main.QueryService", lambda config=None, video_index=None, warmup=False: service)

    # top_k=2 is rejected
    test_args = [
        "main.py", "search", "test",
        "--config", str(synthetic_cli_config_file),
        "--top-k", "2",
        "--json",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "error"
    assert "between 3 and 5" in data["message"]

    # top_k=3 is accepted
    test_args = [
        "main.py", "search", "test",
        "--config", str(synthetic_cli_config_file),
        "--top-k", "3",
        "--json",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] in ("ok", "no_match")
