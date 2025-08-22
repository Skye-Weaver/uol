import os
from pathlib import Path

from Components import config as cfg_module
from Components.config import AppConfig, PathsConfig, ProcessingConfig, LLMConfig, LoggingConfig
from Components.Paths import resolve_path, fonts_path


def _make_app_config(base_dir: Path, fonts_dir: str = "fonts") -> AppConfig:
    return AppConfig(
        processing=ProcessingConfig(),
        llm=LLMConfig(),
        logging=LoggingConfig(),
        paths=PathsConfig(
            base_dir=str(Path(base_dir).resolve()),
            fonts_dir=fonts_dir,
        ),
    )


def test_fonts_path_points_to_tmp_font(tmp_path, monkeypatch):
    # Arrange: create fonts dir and dummy font file under temporary base_dir
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    font_file = fonts_dir / "Montserrat-Bold.ttf"
    font_file.write_text("dummy ttf bytes", encoding="utf-8")

    # Inject temporary config
    app_cfg = _make_app_config(tmp_path, "fonts")
    # Ensure get_config() returns our injected config
    monkeypatch.setattr(cfg_module, "_CONFIG", app_cfg, raising=False)

    # Act
    fp = fonts_path("Montserrat-Bold.ttf")

    # Assert
    assert Path(fp).resolve() == font_file.resolve()
    assert os.path.isabs(fp), "fonts_path must return an absolute path"


def test_resolve_path_builds_from_base_dir(tmp_path, monkeypatch):
    # Inject temporary config with base_dir at tmp_path
    app_cfg = _make_app_config(tmp_path, "fonts")
    monkeypatch.setattr(cfg_module, "_CONFIG", app_cfg, raising=False)

    # Act
    rp = resolve_path("a", "b", "c.txt")

    # Assert
    expected = (tmp_path / "a" / "b" / "c.txt").resolve()
    assert Path(rp).resolve() == expected
    assert os.path.isabs(rp), "resolve_path must return an absolute path"