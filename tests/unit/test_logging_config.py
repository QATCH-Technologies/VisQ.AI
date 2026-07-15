import logging

import pytest
from loguru import logger as loguru_logger

from visqai import logging_config


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Each test gets a clean, unconfigured state and leaves loguru without
    dangling sinks afterward -- configure_logging()'s module-level
    `_configured` flag would otherwise leak between tests."""
    logging_config._configured = False
    yield
    loguru_logger.remove()
    logging_config._configured = False


def test_configure_logging_creates_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    assert not log_dir.exists()
    logging_config.configure_logging(log_dir=log_dir, force=True)
    assert log_dir.exists()


def test_configure_logging_writes_to_file(tmp_path):
    log_dir = tmp_path / "logs"
    logging_config.configure_logging(log_dir=log_dir, level="DEBUG", file_level="DEBUG", force=True)
    loguru_logger.info("hello from test")
    loguru_logger.complete()  # flush the enqueued file sink before reading it back

    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text(encoding="utf-8")
    assert "hello from test" in content
    assert "INFO" in content


def test_configure_logging_respects_file_level_below_console(tmp_path):
    """file_level defaults more verbose (DEBUG) than console (INFO) -- a
    DEBUG message should land in the file even though it wouldn't print."""
    log_dir = tmp_path / "logs"
    logging_config.configure_logging(log_dir=log_dir, level="INFO", file_level="DEBUG", force=True)
    loguru_logger.debug("debug-only message")
    loguru_logger.complete()

    content = list(log_dir.glob("*.log"))[0].read_text(encoding="utf-8")
    assert "debug-only message" in content


def test_configure_logging_is_idempotent_without_force(tmp_path):
    log_dir_a = tmp_path / "a"
    log_dir_b = tmp_path / "b"
    logging_config.configure_logging(log_dir=log_dir_a, force=True)
    logging_config.configure_logging(log_dir=log_dir_b)  # no force -> no-op
    assert log_dir_a.exists()
    assert not log_dir_b.exists()


def test_configure_logging_force_reconfigures(tmp_path):
    log_dir_a = tmp_path / "a"
    log_dir_b = tmp_path / "b"
    logging_config.configure_logging(log_dir=log_dir_a, force=True)
    logging_config.configure_logging(log_dir=log_dir_b, force=True)
    assert log_dir_b.exists()


def test_stdlib_logging_is_routed_into_loguru_file_sink(tmp_path):
    """The InterceptHandler is the mechanism that lets every existing
    logging.getLogger(...) call site across the codebase land in the same
    log file without being rewritten to use loguru directly."""
    log_dir = tmp_path / "logs"
    logging_config.configure_logging(log_dir=log_dir, level="DEBUG", file_level="DEBUG", force=True)

    stdlib_logger = logging.getLogger("some.arbitrary.module")
    stdlib_logger.info("captured via stdlib logging")
    loguru_logger.complete()

    content = list(log_dir.glob("*.log"))[0].read_text(encoding="utf-8")
    assert "captured via stdlib logging" in content


def test_get_logger_configures_if_needed(tmp_path, monkeypatch):
    monkeypatch.setenv("VISQAI_LOG_DIR", str(tmp_path / "envlogs"))
    lg = logging_config.get_logger("mymodule")
    lg.info("via get_logger")
    loguru_logger.complete()
    assert (tmp_path / "envlogs").exists()


def test_default_log_dir_resolves_under_repo_root(monkeypatch):
    monkeypatch.delenv("VISQAI_LOG_DIR", raising=False)
    log_dir = logging_config._default_log_dir()
    assert log_dir.name == "logs"
