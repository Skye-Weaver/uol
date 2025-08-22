from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import os

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


@dataclass
class ProcessingConfig:
    use_animated_captions: bool = True
    shorts_dir: str = "shorts"
    videos_dir: str = "videos"
    crop_bottom_percent: float = 0.0
    min_video_dimension_px: int = 100
    log_transcription_preview_len: int = 200


@dataclass
class LLMConfig:
    model_name: str = "gemini-2.5-flash"
    temperature_highlights: float = 0.2
    temperature_metadata: float = 1.0
    max_attempts_highlights: int = 3
    max_attempts_metadata: int = 3
    highlight_min_sec: int = 29
    highlight_max_sec: int = 61
    max_highlights: int = 20


@dataclass
class AppConfig:
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def _as_bool(v: Any, default: bool) -> bool:
    try:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off"):
                return False
    except Exception:
        pass
    return default


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_str(v: Any, default: str) -> str:
    try:
        return str(v)
    except Exception:
        return default


def _clamp(val: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(val)))
    except Exception:
        return lo


def load_config(path: str = "config.yaml") -> AppConfig:
    """
    Загружает конфигурацию приложения из YAML-файла и накладывает её на значения по умолчанию.

    Поведение:
    - Если файл отсутствует или не читается, возвращает значения по умолчанию и печатает:
      "Конфиг не найден. Использую значения по умолчанию."
    - Если файл частично заполнен, недостающие параметры берутся из дефолтов.
    - Выполняется базовая валидация типов и диапазонов (температуры, проценты, интервалы и т.п.).

    Возвращает:
    - Объект AppConfig с заполненными секциями processing и llm.
    """
    defaults = AppConfig()

    if not os.path.exists(path):
        print("Конфиг не найден. Использую значения по умолчанию.")
        return defaults

    data: Dict[str, Any] = {}
    try:
        if yaml is None:
            raise RuntimeError("PyYAML не установлен")
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)  # type: ignore
            if isinstance(loaded, dict):
                data = loaded
    except Exception:
        # В случае любой ошибки парсинга — мягко откатываемся к дефолтам
        return defaults

    p_in = data.get("processing", {}) or {}
    l_in = data.get("llm", {}) or {}
    if not isinstance(p_in, dict):
        p_in = {}
    if not isinstance(l_in, dict):
        l_in = {}

    # Processing
    p = ProcessingConfig(
        use_animated_captions=_as_bool(
            p_in.get("use_animated_captions", defaults.processing.use_animated_captions),
            defaults.processing.use_animated_captions,
        ),
        shorts_dir=_as_str(p_in.get("shorts_dir", defaults.processing.shorts_dir), defaults.processing.shorts_dir),
        videos_dir=_as_str(p_in.get("videos_dir", defaults.processing.videos_dir), defaults.processing.videos_dir),
        crop_bottom_percent=_clamp(
            _as_float(p_in.get("crop_bottom_percent", defaults.processing.crop_bottom_percent),
                      defaults.processing.crop_bottom_percent),
            0.0,
            100.0,
        ),
        min_video_dimension_px=max(
            1,
            _as_int(p_in.get("min_video_dimension_px", defaults.processing.min_video_dimension_px),
                    defaults.processing.min_video_dimension_px),
        ),
        log_transcription_preview_len=max(
            1,
            _as_int(p_in.get("log_transcription_preview_len", defaults.processing.log_transcription_preview_len),
                    defaults.processing.log_transcription_preview_len),
        ),
    )

    # LLM
    t_h = _as_float(l_in.get("temperature_highlights", defaults.llm.temperature_highlights),
                    defaults.llm.temperature_highlights)
    t_m = _as_float(l_in.get("temperature_metadata", defaults.llm.temperature_metadata),
                    defaults.llm.temperature_metadata)
    t_h = _clamp(t_h, 0.0, 2.0)
    t_m = _clamp(t_m, 0.0, 2.0)

    max_att_h = max(1, _as_int(l_in.get("max_attempts_highlights", defaults.llm.max_attempts_highlights),
                               defaults.llm.max_attempts_highlights))
    max_att_m = max(1, _as_int(l_in.get("max_attempts_metadata", defaults.llm.max_attempts_metadata),
                               defaults.llm.max_attempts_metadata))

    h_min = _as_int(l_in.get("highlight_min_sec", defaults.llm.highlight_min_sec),
                    defaults.llm.highlight_min_sec)
    h_max = _as_int(l_in.get("highlight_max_sec", defaults.llm.highlight_max_sec),
                    defaults.llm.highlight_max_sec)
    if h_min < 0:
        h_min = defaults.llm.highlight_min_sec
    if h_max <= h_min:
        h_max = defaults.llm.highlight_max_sec

    max_hls = max(1, _as_int(l_in.get("max_highlights", defaults.llm.max_highlights),
                             defaults.llm.max_highlights))

    l = LLMConfig(
        model_name=_as_str(l_in.get("model_name", defaults.llm.model_name), defaults.llm.model_name),
        temperature_highlights=t_h,
        temperature_metadata=t_m,
        max_attempts_highlights=max_att_h,
        max_attempts_metadata=max_att_m,
        highlight_min_sec=h_min,
        highlight_max_sec=h_max,
        max_highlights=max_hls,
    )

    return AppConfig(processing=p, llm=l)


_CONFIG: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Ленивая кэширующая обёртка над load_config(). Загружает конфигурацию один раз
    из файла (по умолчанию config.yaml), кеширует результат в памяти и возвращает
    один и тот же экземпляр AppConfig при последующих вызовах.
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG