"""Tests for config loading and system instruction composition."""

import os
from pathlib import Path

import pytest

from gpal.server import (
    _build_system_instruction,
    _compose_instruction,
    _SYSTEM_AGENT,
    _load_config,
)


class TestLoadConfig:
    """Tests for _load_config()."""

    def test_no_config_file(self, tmp_path, monkeypatch):
        """Returns empty dict when no config file exists."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert _load_config() == {}

    def test_valid_config(self, tmp_path, monkeypatch):
        """Parses valid TOML config."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_dir = tmp_path / "gpal"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            'system_prompts = ["~/GEMINI.md"]\ninclude_default_prompt = false\n'
        )
        config = _load_config()
        assert config["system_prompts"] == ["~/GEMINI.md"]
        assert config["include_default_prompt"] is False

    def test_invalid_toml(self, tmp_path, monkeypatch):
        """Returns empty dict on invalid TOML (non-fatal)."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_dir = tmp_path / "gpal"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text("this is not valid [[[toml")
        assert _load_config() == {}

    def test_xdg_default(self, tmp_path, monkeypatch):
        """Falls back to ~/.config when XDG_CONFIG_HOME is unset."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        config_dir = tmp_path / ".config" / "gpal"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text('system_prompt = "hello"\n')
        config = _load_config()
        assert config["system_prompt"] == "hello"


class TestBuildSystemInstruction:
    """Tests for _build_system_instruction()."""

    def test_default_only(self):
        """With empty config, returns empty user layers and built-in source."""
        text, sources = _build_system_instruction({})
        # _build_system_instruction returns only user layers; the role prompt
        # is composed separately via _compose_instruction
        assert text == ""
        assert sources == ["built-in (role-specific)"]

    def test_compose_instruction(self):
        """_compose_instruction combines role prompt with user layers."""
        full = _compose_instruction(_SYSTEM_AGENT)
        assert "consultant AI" in full
        assert "Look at the code first" in full

    def test_no_default(self, tmp_path):
        """--no-default-prompt suppresses the built-in source label."""
        prompt_file = tmp_path / "custom.md"
        prompt_file.write_text("Custom instruction")
        text, sources = _build_system_instruction(
            {}, cli_prompt_files=[str(prompt_file)], no_default=True
        )
        assert "Custom instruction" in text
        assert "built-in" not in sources[0] if sources else True

    def test_no_default_config(self, tmp_path):
        """include_default_prompt=false in config suppresses built-in source."""
        prompt_file = tmp_path / "custom.md"
        prompt_file.write_text("Custom instruction")
        config = {
            "include_default_prompt": False,
            "system_prompts": [str(prompt_file)],
        }
        text, sources = _build_system_instruction(config)
        assert "Custom instruction" in text
        assert all("built-in" not in s for s in sources)

    def test_composition_order(self, tmp_path):
        """Verifies correct ordering: config files, inline, CLI files."""
        config_file = tmp_path / "config-prompt.md"
        config_file.write_text("FROM_CONFIG_FILE")
        cli_file = tmp_path / "cli-prompt.md"
        cli_file.write_text("FROM_CLI")

        config = {
            "system_prompts": [str(config_file)],
            "system_prompt": "FROM_INLINE",
        }
        text, sources = _build_system_instruction(
            config, cli_prompt_files=[str(cli_file)]
        )

        # User layers are ordered: config files, inline, CLI files
        config_pos = text.index("FROM_CONFIG_FILE")
        inline_pos = text.index("FROM_INLINE")
        cli_pos = text.index("FROM_CLI")
        assert config_pos < inline_pos < cli_pos

        # Sources: built-in label first, then user layers
        assert sources[0] == "built-in (role-specific)"
        assert str(config_file) in sources[1]
        assert "config.toml (inline)" in sources[2]
        assert "cli-prompt" in sources[3]

    def test_missing_prompt_file(self, tmp_path):
        """Missing prompt files are warned about but don't crash."""
        config = {"system_prompts": [str(tmp_path / "nonexistent.md")]}
        text, sources = _build_system_instruction(config)
        assert text == ""
        assert sources == ["built-in (role-specific)"]

    def test_fallback_when_all_suppressed(self):
        """If no_default and no files, returns empty layers."""
        text, sources = _build_system_instruction({}, no_default=True)
        assert text == ""
        assert sources == []

    def test_binary_prompt_file(self, tmp_path):
        """Binary file as prompt logs warning, doesn't crash."""
        binary_file = tmp_path / "binary.md"
        binary_file.write_bytes(b"\x80\x81\x82\xff\xfe")
        config = {"system_prompts": [str(binary_file)]}
        text, sources = _build_system_instruction(config)
        assert text == ""

    def test_system_prompts_wrong_type(self):
        """String instead of list for system_prompts is handled gracefully."""
        config = {"system_prompts": "not-a-list.md"}
        text, sources = _build_system_instruction(config)
        assert text == ""
        assert sources == ["built-in (role-specific)"]

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        """Tilde in paths gets expanded."""
        monkeypatch.setenv("HOME", str(tmp_path))
        prompt_file = tmp_path / "GEMINI.md"
        prompt_file.write_text("Tilde expanded")
        config = {"system_prompts": ["~/GEMINI.md"]}
        text, sources = _build_system_instruction(config)
        assert "Tilde expanded" in text

    def test_envvar_expansion(self, tmp_path, monkeypatch):
        """Environment variables in paths get expanded."""
        monkeypatch.setenv("WORKSPACE", str(tmp_path))
        prompt_file = tmp_path / "CLAUDE.md"
        prompt_file.write_text("Workspace prompt")
        config = {"system_prompts": ["$WORKSPACE/CLAUDE.md"]}
        text, sources = _build_system_instruction(config)
        assert "Workspace prompt" in text

    def test_envvar_expansion_cli(self, tmp_path, monkeypatch):
        """Environment variables work in CLI --system-prompt paths too."""
        monkeypatch.setenv("PROJECT", str(tmp_path))
        prompt_file = tmp_path / "context.md"
        prompt_file.write_text("CLI envvar prompt")
        text, sources = _build_system_instruction(
            {}, cli_prompt_files=["$PROJECT/context.md"]
        )
        assert "CLI envvar prompt" in text
