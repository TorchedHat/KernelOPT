"""Tests for kernelopt.config — config file I/O."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestConfigPath:
    def test_returns_path_ending_in_config_toml(self):
        from kernelopt.config import config_path
        p = config_path()
        assert p.name == "config.toml"
        assert p.parent.name == "kernelopt"
        assert ".config" in str(p)


class TestLoadConfig:
    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        from kernelopt.config import load_config
        missing = tmp_path / "config.toml"
        with patch("kernelopt.config.config_path", return_value=missing):
            result = load_config()
        assert result == {}

    def test_returns_parsed_toml(self, tmp_path):
        from kernelopt.config import load_config
        cfg = tmp_path / "config.toml"
        cfg.write_text('[vertex]\nproject_id = "my-project"\nregion = "us-east5"\n')
        with patch("kernelopt.config.config_path", return_value=cfg):
            result = load_config()
        assert result["vertex"]["project_id"] == "my-project"
        assert result["vertex"]["region"] == "us-east5"

    def test_returns_empty_dict_on_malformed_toml(self, tmp_path):
        from kernelopt.config import load_config
        cfg = tmp_path / "config.toml"
        cfg.write_text("not valid = = toml ===")
        with patch("kernelopt.config.config_path", return_value=cfg):
            result = load_config()
        assert result == {}


class TestSaveConfig:
    def test_writes_project_id_and_region(self, tmp_path):
        from kernelopt.config import save_config
        target = tmp_path / "kernelopt" / "config.toml"
        with patch("kernelopt.config.config_path", return_value=target):
            path = save_config("my-project", "us-east5")
        assert path == target
        content = target.read_text()
        assert "my-project" in content
        assert "us-east5" in content

    def test_creates_parent_directories(self, tmp_path):
        from kernelopt.config import save_config
        target = tmp_path / "deep" / "nested" / "config.toml"
        with patch("kernelopt.config.config_path", return_value=target):
            save_config("proj", "us-east5")
        assert target.exists()

    def test_roundtrip_load_after_save(self, tmp_path):
        from kernelopt.config import load_config, save_config
        target = tmp_path / "config.toml"
        with patch("kernelopt.config.config_path", return_value=target):
            save_config("roundtrip-project", "europe-west4")
            result = load_config()
        assert result["vertex"]["project_id"] == "roundtrip-project"
        assert result["vertex"]["region"] == "europe-west4"
