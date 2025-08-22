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
class LoggingConfig:
    # Основные настройки логирования
    log_dir: str = "logs"
    log_level: str = "INFO"
    enable_console_logging: bool = True
    enable_file_logging: bool = True

    # Настройки ротации логов
    log_rotation_max_bytes: int = 10485760  # 10MB
    log_rotation_backup_count: int = 5
    log_rotation_when: str = "midnight"
    log_compression: bool = False

    # Отдельные логгеры для разных типов сообщений
    enable_main_logger: bool = True
    enable_performance_logger: bool = True
    enable_error_logger: bool = True
    enable_debug_logger: bool = False

    # Настройки производительности
    enable_performance_monitoring: bool = True
    performance_log_max_bytes: int = 5242880  # 5MB
    performance_log_backup_count: int = 3
    performance_monitoring_interval: float = 0.5

    # Настройки GPU мониторинга
    enable_gpu_monitoring: bool = True
    gpu_priority_mode: bool = True
    gpu_memory_threshold: float = 0.9
    gpu_temperature_threshold: int = 80

    # Настройки CPU и памяти
    enable_cpu_monitoring: bool = True
    enable_memory_monitoring: bool = True
    memory_threshold: float = 0.85
    cpu_threshold: float = 90.0

    # Настройки прогресс-баров
    enable_progress_bars: bool = True
    progress_bar_update_interval: float = 0.1

    # Системная информация
    enable_system_info_logging: bool = True
    system_info_log_interval: int = 3600

    # Асинхронная обработка
    enable_async_logging: bool = True
    log_queue_size: int = 1000
    log_worker_threads: int = 2

    # Форматирование логов
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    enable_colors: bool = True

    # Фильтры логирования
    log_filters: list = field(default_factory=lambda: [
        "urllib3.connectionpool",
        "PIL.PngImagePlugin"
    ])

    # Настройки для разных сред
    development_mode: bool = False
    enable_detailed_tracing: bool = False
    enable_function_call_logging: bool = False

    # Ресурсный мониторинг
    resource_monitoring_interval: float = 1.0
    enable_resource_monitoring: bool = True
    enable_resource_alerts: bool = True
    alert_threshold_duration: float = 30.0


@dataclass
class AppConfig:
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


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
    - Объект AppConfig с заполненными секциями processing, llm и logging.
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
    log_in = data.get("logging", {}) or {}
    if not isinstance(p_in, dict):
        p_in = {}
    if not isinstance(l_in, dict):
        l_in = {}
    if not isinstance(log_in, dict):
        log_in = {}

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

    # Logging
    log = LoggingConfig(
        # Основные настройки логирования
        log_dir=_as_str(log_in.get("log_dir", defaults.logging.log_dir), defaults.logging.log_dir),
        log_level=_as_str(log_in.get("log_level", defaults.logging.log_level), defaults.logging.log_level),
        enable_console_logging=_as_bool(
            log_in.get("enable_console_logging", defaults.logging.enable_console_logging),
            defaults.logging.enable_console_logging,
        ),
        enable_file_logging=_as_bool(
            log_in.get("enable_file_logging", defaults.logging.enable_file_logging),
            defaults.logging.enable_file_logging,
        ),

        # Настройки ротации логов
        log_rotation_max_bytes=max(
            1024,
            _as_int(log_in.get("log_rotation_max_bytes", defaults.logging.log_rotation_max_bytes),
                    defaults.logging.log_rotation_max_bytes),
        ),
        log_rotation_backup_count=max(
            1,
            _as_int(log_in.get("log_rotation_backup_count", defaults.logging.log_rotation_backup_count),
                    defaults.logging.log_rotation_backup_count),
        ),
        log_rotation_when=_as_str(log_in.get("log_rotation_when", defaults.logging.log_rotation_when),
                                  defaults.logging.log_rotation_when),
        log_compression=_as_bool(
            log_in.get("log_compression", defaults.logging.log_compression),
            defaults.logging.log_compression,
        ),

        # Отдельные логгеры для разных типов сообщений
        enable_main_logger=_as_bool(
            log_in.get("enable_main_logger", defaults.logging.enable_main_logger),
            defaults.logging.enable_main_logger,
        ),
        enable_performance_logger=_as_bool(
            log_in.get("enable_performance_logger", defaults.logging.enable_performance_logger),
            defaults.logging.enable_performance_logger,
        ),
        enable_error_logger=_as_bool(
            log_in.get("enable_error_logger", defaults.logging.enable_error_logger),
            defaults.logging.enable_error_logger,
        ),
        enable_debug_logger=_as_bool(
            log_in.get("enable_debug_logger", defaults.logging.enable_debug_logger),
            defaults.logging.enable_debug_logger,
        ),

        # Настройки производительности
        enable_performance_monitoring=_as_bool(
            log_in.get("enable_performance_monitoring", defaults.logging.enable_performance_monitoring),
            defaults.logging.enable_performance_monitoring,
        ),
        performance_log_max_bytes=max(
            1024,
            _as_int(log_in.get("performance_log_max_bytes", defaults.logging.performance_log_max_bytes),
                    defaults.logging.performance_log_max_bytes),
        ),
        performance_log_backup_count=max(
            1,
            _as_int(log_in.get("performance_log_backup_count", defaults.logging.performance_log_backup_count),
                    defaults.logging.performance_log_backup_count),
        ),
        performance_monitoring_interval=max(
            0.1,
            _as_float(log_in.get("performance_monitoring_interval", defaults.logging.performance_monitoring_interval),
                      defaults.logging.performance_monitoring_interval),
        ),

        # Настройки GPU мониторинга
        enable_gpu_monitoring=_as_bool(
            log_in.get("enable_gpu_monitoring", defaults.logging.enable_gpu_monitoring),
            defaults.logging.enable_gpu_monitoring,
        ),
        gpu_priority_mode=_as_bool(
            log_in.get("gpu_priority_mode", defaults.logging.gpu_priority_mode),
            defaults.logging.gpu_priority_mode,
        ),
        gpu_memory_threshold=_clamp(
            _as_float(log_in.get("gpu_memory_threshold", defaults.logging.gpu_memory_threshold),
                      defaults.logging.gpu_memory_threshold),
            0.1, 1.0,
        ),
        gpu_temperature_threshold=max(
            1,
            _as_int(log_in.get("gpu_temperature_threshold", defaults.logging.gpu_temperature_threshold),
                    defaults.logging.gpu_temperature_threshold),
        ),

        # Настройки CPU и памяти
        enable_cpu_monitoring=_as_bool(
            log_in.get("enable_cpu_monitoring", defaults.logging.enable_cpu_monitoring),
            defaults.logging.enable_cpu_monitoring,
        ),
        enable_memory_monitoring=_as_bool(
            log_in.get("enable_memory_monitoring", defaults.logging.enable_memory_monitoring),
            defaults.logging.enable_memory_monitoring,
        ),
        memory_threshold=_clamp(
            _as_float(log_in.get("memory_threshold", defaults.logging.memory_threshold),
                      defaults.logging.memory_threshold),
            0.1, 1.0,
        ),
        cpu_threshold=_clamp(
            _as_float(log_in.get("cpu_threshold", defaults.logging.cpu_threshold),
                      defaults.logging.cpu_threshold),
            1.0, 100.0,
        ),

        # Настройки прогресс-баров
        enable_progress_bars=_as_bool(
            log_in.get("enable_progress_bars", defaults.logging.enable_progress_bars),
            defaults.logging.enable_progress_bars,
        ),
        progress_bar_update_interval=max(
            0.01,
            _as_float(log_in.get("progress_bar_update_interval", defaults.logging.progress_bar_update_interval),
                      defaults.logging.progress_bar_update_interval),
        ),

        # Системная информация
        enable_system_info_logging=_as_bool(
            log_in.get("enable_system_info_logging", defaults.logging.enable_system_info_logging),
            defaults.logging.enable_system_info_logging,
        ),
        system_info_log_interval=max(
            60,
            _as_int(log_in.get("system_info_log_interval", defaults.logging.system_info_log_interval),
                    defaults.logging.system_info_log_interval),
        ),

        # Асинхронная обработка
        enable_async_logging=_as_bool(
            log_in.get("enable_async_logging", defaults.logging.enable_async_logging),
            defaults.logging.enable_async_logging,
        ),
        log_queue_size=max(
            10,
            _as_int(log_in.get("log_queue_size", defaults.logging.log_queue_size),
                    defaults.logging.log_queue_size),
        ),
        log_worker_threads=max(
            1,
            _as_int(log_in.get("log_worker_threads", defaults.logging.log_worker_threads),
                    defaults.logging.log_worker_threads),
        ),

        # Форматирование логов
        log_format=_as_str(log_in.get("log_format", defaults.logging.log_format), defaults.logging.log_format),
        log_date_format=_as_str(log_in.get("log_date_format", defaults.logging.log_date_format),
                                defaults.logging.log_date_format),
        enable_colors=_as_bool(
            log_in.get("enable_colors", defaults.logging.enable_colors),
            defaults.logging.enable_colors,
        ),

        # Фильтры логирования
        log_filters=log_in.get("log_filters", defaults.logging.log_filters) if isinstance(log_in.get("log_filters"), list) else defaults.logging.log_filters,

        # Настройки для разных сред
        development_mode=_as_bool(
            log_in.get("development_mode", defaults.logging.development_mode),
            defaults.logging.development_mode,
        ),
        enable_detailed_tracing=_as_bool(
            log_in.get("enable_detailed_tracing", defaults.logging.enable_detailed_tracing),
            defaults.logging.enable_detailed_tracing,
        ),
        enable_function_call_logging=_as_bool(
            log_in.get("enable_function_call_logging", defaults.logging.enable_function_call_logging),
            defaults.logging.enable_function_call_logging,
        ),

        # Ресурсный мониторинг
        resource_monitoring_interval=max(
            0.1,
            _as_float(log_in.get("resource_monitoring_interval", defaults.logging.resource_monitoring_interval),
                      defaults.logging.resource_monitoring_interval),
        ),
        enable_resource_monitoring=_as_bool(
            log_in.get("enable_resource_monitoring", defaults.logging.enable_resource_monitoring),
            defaults.logging.enable_resource_monitoring,
        ),
        enable_resource_alerts=_as_bool(
            log_in.get("enable_resource_alerts", defaults.logging.enable_resource_alerts),
            defaults.logging.enable_resource_alerts,
        ),
        alert_threshold_duration=max(
            1.0,
            _as_float(log_in.get("alert_threshold_duration", defaults.logging.alert_threshold_duration),
                      defaults.logging.alert_threshold_duration),
        ),
    )

    return AppConfig(processing=p, llm=l, logging=log)


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