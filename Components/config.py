from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import os
import re
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


@dataclass
class ProcessingConfig:
    use_animated_captions: bool = True
    shorts_dir: str = "shorts"
    videos_dir: str = "videos"
    transcriptions_dir: str = "transcriptions"
    crop_bottom_percent: float = 0.0
    min_video_dimension_px: int = 100
    log_transcription_preview_len: int = 200
    crop_mode: str = "70_percent_blur"  # "average_face" or "70_percent_blur"


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
class IntelligentPauseAnalysisConfig:
    """Конфигурация для интеллектуального анализа пауз"""
    enabled: bool = True  # Включить ИИ-анализ пауз
    model: str = "gemini-2.5-flash-lite"  # Модель для простых задач анализа пауз
    temperature: float = 0.1  # Температура для анализа пауз (низкая для консистентности)
    max_attempts: int = 2  # Максимум попыток для анализа пауз
    auto_trim_confidence_threshold: float = 0.8  # Порог уверенности для автоматической обрезки
    batch_size: int = 10  # Размер батча для пакетного анализа пауз
    cache_enabled: bool = True  # Кеширование результатов анализа пауз
    cache_ttl_hours: int = 24  # Время жизни кеша в часах

    # Категории пауз для классификации
    pause_categories: dict = field(default_factory=lambda: {
        "structural": ["sentence_end", "paragraph_break", "topic_change"],  # Структурные паузы (не обрезать)
        "filler": ["um", "uh", "er", "ah", "like", "you_know"],  # Заполнители (обрезать)
        "emphasis": ["dramatic_pause", "for_effect"],  # Паузы для эффекта (анализировать контекст)
        "breathing": ["breath", "inhale", "exhale"]  # Дыхательные паузы (обрезать при длинных)
    })

    # Весовые коэффициенты для определения важности пауз
    importance_weights: dict = field(default_factory=lambda: {
        "duration": -0.4,  # Чем длиннее пауза, тем менее важна (отрицательный вес)
        "position": 0.3,   # Позиция в предложении (начало/середина/конец)
        "context": 0.4,    # Контекст вокруг паузы
        "audio_features": 0.2,  # Аудио-характеристики (тишина vs шум)
        "linguistic": 0.3  # Лингвистический анализ
    })

    # Оптимизация API
    api_optimization: dict = field(default_factory=lambda: {
        "use_batch_processing": True,  # Использовать пакетную обработку
        "max_concurrent_requests": 3,  # Максимум одновременных запросов
        "rate_limit_delay": 1.0,  # Задержка между запросами (секунды)
        "retry_on_failure": True,  # Повторять при неудачах
        "fallback_to_legacy": True  # Откат на старую логику при ошибках ИИ
    })

    def get(self, key: str, default=None):
        """
        Метод для совместимости с кодом, который использует объект как словарь.
        Возвращает значение атрибута по имени или значение по умолчанию.
        """
        if hasattr(self, key):
            return getattr(self, key)
        return default


@dataclass
class FilmModeConfig:
    """Конфигурация для режима 'фильм'"""
    enabled: bool = True
    combo_duration: list = field(default_factory=lambda: [10, 20])  # секунды для COMBO моментов
    single_duration: list = field(default_factory=lambda: [30, 60])  # секунды для SINGLE моментов
    max_moments: int = 50  # монолитный режим: максимум моментов для анализа (увеличено для Film Mode v2)
    pause_threshold: float = 0.7  # порог для определения длинных пауз (секунды)
    filler_words: list = field(default_factory=lambda: ["э-э", "м-м", "ну", "эээ", "гм", "кхм"])  # слова-заполнители
    min_quality_score: float = 0.5  # минимальный порог качества для включения момента
    generate_shorts: bool = True  # генерировать шорты из найденных моментов

    # Интеллектуальный анализ пауз
    intelligent_pause_analysis: IntelligentPauseAnalysisConfig = field(default_factory=IntelligentPauseAnalysisConfig)

    # Эталонные ключевые слова для подсчета совпадений (новая система ранжирования)
    reference_keywords: dict = field(default_factory=lambda: {
        'emotional_peaks': ["эмоции", "чувства", "переживания", "радость", "гнев", "страх", "удивление", "грусть", "любовь", "ненависть", "восторг", "отчаяние", "надежда", "разочарование", "триумф", "поражение"],
        'conflict_escalation': ["конфликт", "спор", "ссора", "драка", "ругань", "оскорбление", "критика", "давление", "напряжение", "эскалация", "столкновение", "противостояние", "борьба", "конкуренция"],
        'punchlines_wit': ["юмор", "шутка", "сарказм", "ирония", "остроумие", "панчлайн", "каламбур", "смех", "комедия", "прикол", "насмешка", "сатира", "пародия", "абсурд"],
        'quotability_memes': ["цитата", "афоризм", "крылатая фраза", "мем", "вирусный", "тренд", "хайп", "легендарный", "знаменитый", "классика", "запоминающийся", "уникальный"],
        'stakes_goals': ["ставки", "цель", "риск", "опасность", "выбор", "решение", "судьба", "жизнь", "смерть", "выигрыш", "проигрыш", "успех", "провал", "достижение", "амбиции"],
        'hooks_cliffhangers': ["вопрос", "загадка", "тайна", "сюрприз", "интрига", "недосказанность", "продолжение", "развязка", "поворот", "неожиданно", "вдруг", "что дальше", "как же так"]
    })

    # Оконный сбор кандидатов (LLM-sweep)
    window_minutes: int = 8   # Уменьшен для большего числа окон
    window_overlap_minutes: int = 2  # Уменьшен для лучшего покрытия
    max_moments_per_window: int = 10  # Увеличено для большего числа моментов на окно

    # Цели и лимиты генерации
    target_shorts_count: int = 30
    generator_top_k: int = 30

    # Дедупликация/диверсификация
    dedupe_iou_threshold: float = 0.5
    diversity_bucket_minutes: int = 5
    min_combo_segments: int = 2
    max_combo_segments: int = 4

    # Весовые коэффициенты для ранжирования моментов
    ranking_weights: dict = field(default_factory=lambda: {
        'emotional_peaks': 0.20,      # Эмоциональные пики и переломы статуса
        'conflict_escalation': 0.18,  # Конфликт и эскалация
        'punchlines_wit': 0.16,       # Панчлайны и остроумие
        'quotability_memes': 0.14,    # Цитатность/мемность
        'stakes_goals': 0.12,         # Ставки и цель
        'hooks_cliffhangers': 0.10,   # Крючки/клиффхэнгеры
        'visual_penalty': -0.10,      # Штраф за визуальную зависимость
        'pace_score': 0.08,           # Плотность речи (слова/сек)
        'silence_penalty': -0.08,     # Наказание за долю длинных пауз
        'diversity_bonus': 0.05       # Бонус за диверсификацию (может заполняться в ранжировании)
    })

    # Пороговые настройки ранжирования и fallback
    ranking: dict = field(default_factory=lambda: {
        'min_quality_threshold': 0.3,  # Уменьшен порог для включения большего числа моментов
        'soft_min_quality': 0.2,       # Уменьшен мягкий порог
        'allow_fallback': True,
        'fallback_top_n': 12,          # Увеличено для гарантии 12+ моментов
        'max_best_moments': 50,        # Увеличено для поддержки большего числа моментов
    })

    # Настройки LLM для анализа фильма
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.3
    llm_max_attempts: int = 3


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
    enable_resource_alerts: bool = True
    alert_threshold_duration: float = 30.0


@dataclass
class PathsConfig:
    """
    Путь-ориентированная конфигурация проекта.

    - base_dir: Абсолютный корень проекта/ресурсов.
    - fonts_dir: Каталог со шрифтами (может быть относительным к base_dir или абсолютным).
    """
    base_dir: str = str(Path(".").resolve())
    fonts_dir: str = "fonts"


@dataclass
class ShadowConfig:
    x_px: int = 2
    y_px: int = 2
    blur_px: int = 1
    color: str = "#00000080"


@dataclass
class AccentPalette:
    urgency: str = "#FFD400"
    drama: str = "#FF3B30"
    positive: str = "#34C759"


@dataclass
class AnimateConfig:
    type: str = "slide-up"  # "pop-in" | "slide-up"
    duration_s: float = 0.35  # [0.2, 0.5]
    easing: str = "easeOutCubic"
    per_word_stagger_ms: int = 120  # [0, 500]


@dataclass
class PositionConfig:
    mode: str = "safe_bottom"  # "safe_bottom" | "center"
    bottom_offset_pct: int = 22  # [0, 100]
    center_offset_pct: int = 12
    boundary_padding_px: int = 10


@dataclass
class EmojiConfig:
    enabled: bool = True
    max_per_short: int = 2  # [0, 5]
    style: str = "shiny"  # "shiny" | "pulse" | "none"


@dataclass
class CaptionsConfig:
    font_size_px: int = 38
    letter_spacing_px: float = 1.5
    line_height: float = 1.3
    base_color: str = "#FFFFFF"
    shadow: ShadowConfig = field(default_factory=ShadowConfig)
    accent_palette: AccentPalette = field(default_factory=AccentPalette)
    animate: AnimateConfig = field(default_factory=AnimateConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    emoji: EmojiConfig = field(default_factory=EmojiConfig)
    strip_punctuation: bool = True
    # New sync/animation fields
    align_to_audio: bool = True
    fade_in_seconds: float = 0.15
    fade_out_seconds: float = 0.12


@dataclass
class AppConfig:
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    captions: CaptionsConfig = field(default_factory=CaptionsConfig)
    film_mode: FilmModeConfig = field(default_factory=FilmModeConfig)


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
    paths_in = data.get("paths", {}) or {}
    film_in = data.get("film_mode", {}) or {}
    if not isinstance(p_in, dict):
        p_in = {}
    if not isinstance(l_in, dict):
        l_in = {}
    if not isinstance(log_in, dict):
        log_in = {}
    if not isinstance(paths_in, dict):
        paths_in = {}
    if not isinstance(film_in, dict):
        film_in = {}

    # Processing
    crop_mode = _as_str(p_in.get("crop_mode", defaults.processing.crop_mode), defaults.processing.crop_mode)
    if crop_mode not in ("average_face", "70_percent_blur"):
        crop_mode = defaults.processing.crop_mode

    p = ProcessingConfig(
        use_animated_captions=_as_bool(
            p_in.get("use_animated_captions", defaults.processing.use_animated_captions),
            defaults.processing.use_animated_captions,
        ),
        shorts_dir=_as_str(p_in.get("shorts_dir", defaults.processing.shorts_dir), defaults.processing.shorts_dir),
        videos_dir=_as_str(p_in.get("videos_dir", defaults.processing.videos_dir), defaults.processing.videos_dir),
        transcriptions_dir=_as_str(
            p_in.get("transcriptions_dir", defaults.processing.transcriptions_dir),
            defaults.processing.transcriptions_dir
        ),
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
        crop_mode=crop_mode,
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

    # Paths
    base_dir_raw = _as_str(paths_in.get("base_dir", "."), ".")
    fonts_dir_raw = _as_str(paths_in.get("fonts_dir", "fonts"), "fonts")

    try:
        base_abs = Path(base_dir_raw).resolve()
    except Exception:
        base_abs = Path(".").resolve()

    paths = PathsConfig(
        base_dir=str(base_abs),
        fonts_dir=fonts_dir_raw,
    )

    # Captions (with safe defaults and clipping)
    captions_in = data.get("captions", {}) or {}
    if not isinstance(captions_in, dict):
        captions_in = {}

    # Local helpers (scoped to load_config)
    def _is_hex_color(s: str) -> bool:
        if not isinstance(s, str):
            return False
        return bool(re.fullmatch(r"#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})", s.strip()))

    def _as_hex_color(v: Any, default: str) -> str:
        s = _as_str(v, default)
        return s if _is_hex_color(s) else default

    def _clamp_int_val(v: Any, default: int, lo: int, hi: int) -> int:
        try:
            iv = int(v)
        except Exception:
            iv = default
        if iv < lo:
            iv = lo
        if iv > hi:
            iv = hi
        return iv

    def _clamp_float_val(v: Any, default: float, lo: float, hi: float) -> float:
        try:
            fv = float(v)
        except Exception:
            fv = default
        return max(lo, min(hi, fv))

    # Top-level caption fields
    fs = _clamp_int_val(
        captions_in.get("font_size_px", defaults.captions.font_size_px),
        defaults.captions.font_size_px, 20, 60
    )
    ls = _clamp_float_val(
        captions_in.get("letter_spacing_px", defaults.captions.letter_spacing_px),
        defaults.captions.letter_spacing_px, 0.0, 10.0
    )
    lh = _clamp_float_val(
        captions_in.get("line_height", defaults.captions.line_height),
        defaults.captions.line_height, 1.0, 2.0
    )
    base_color = _as_hex_color(
        captions_in.get("base_color", defaults.captions.base_color),
        defaults.captions.base_color
    )

    # Shadow
    shadow_in = captions_in.get("shadow", {}) or {}
    if not isinstance(shadow_in, dict):
        shadow_in = {}
    shadow = ShadowConfig(
        x_px=_clamp_int_val(
            shadow_in.get("x_px", defaults.captions.shadow.x_px),
            defaults.captions.shadow.x_px, 0, 20
        ),
        y_px=_clamp_int_val(
            shadow_in.get("y_px", defaults.captions.shadow.y_px),
            defaults.captions.shadow.y_px, 0, 20
        ),
        blur_px=_clamp_int_val(
            shadow_in.get("blur_px", defaults.captions.shadow.blur_px),
            defaults.captions.shadow.blur_px, 0, 16
        ),
        color=_as_hex_color(
            shadow_in.get("color", defaults.captions.shadow.color),
            defaults.captions.shadow.color
        ),
    )

    # Accent palette
    palette_in = captions_in.get("accent_palette", {}) or {}
    if not isinstance(palette_in, dict):
        palette_in = {}
    accent_palette = AccentPalette(
        urgency=_as_hex_color(
            palette_in.get("urgency", defaults.captions.accent_palette.urgency),
            defaults.captions.accent_palette.urgency
        ),
        drama=_as_hex_color(
            palette_in.get("drama", defaults.captions.accent_palette.drama),
            defaults.captions.accent_palette.drama
        ),
        positive=_as_hex_color(
            palette_in.get("positive", defaults.captions.accent_palette.positive),
            defaults.captions.accent_palette.positive
        ),
    )

    # Animate
    animate_in = captions_in.get("animate", {}) or {}
    if not isinstance(animate_in, dict):
        animate_in = {}
    a_type = _as_str(animate_in.get("type", defaults.captions.animate.type), defaults.captions.animate.type)
    if a_type not in ("pop-in", "slide-up"):
        a_type = defaults.captions.animate.type
    duration_s = _clamp_float_val(
        animate_in.get("duration_s", defaults.captions.animate.duration_s),
        defaults.captions.animate.duration_s, 0.2, 0.5
    )
    easing = _as_str(animate_in.get("easing", defaults.captions.animate.easing), defaults.captions.animate.easing)
    per_word_stagger_ms = _clamp_int_val(
        animate_in.get("per_word_stagger_ms", defaults.captions.animate.per_word_stagger_ms),
        defaults.captions.animate.per_word_stagger_ms, 0, 500
    )
    animate = AnimateConfig(
        type=a_type,
        duration_s=duration_s,
        easing=easing,
        per_word_stagger_ms=per_word_stagger_ms,
    )

    # Position
    position_in = captions_in.get("position", {}) or {}
    if not isinstance(position_in, dict):
        position_in = {}
    mode = _as_str(position_in.get("mode", defaults.captions.position.mode), defaults.captions.position.mode)
    if mode not in ("safe_bottom", "center"):
        mode = defaults.captions.position.mode
    bottom_offset_pct = _clamp_int_val(
        position_in.get("bottom_offset_pct", defaults.captions.position.bottom_offset_pct),
        defaults.captions.position.bottom_offset_pct, 0, 100
    )
    center_offset_pct = _clamp_int_val(
        position_in.get("center_offset_pct", defaults.captions.position.center_offset_pct),
        defaults.captions.position.center_offset_pct, -50, 50
    )
    boundary_padding_px = _clamp_int_val(
        position_in.get("boundary_padding_px", defaults.captions.position.boundary_padding_px),
        defaults.captions.position.boundary_padding_px, 0, 100
    )
    position = PositionConfig(
        mode=mode,
        bottom_offset_pct=bottom_offset_pct,
        center_offset_pct=center_offset_pct,
        boundary_padding_px=boundary_padding_px
    )

    # Emoji
    emoji_in = captions_in.get("emoji", {}) or {}
    if not isinstance(emoji_in, dict):
        emoji_in = {}
    enabled = _as_bool(emoji_in.get("enabled", defaults.captions.emoji.enabled), defaults.captions.emoji.enabled)
    max_per_short = _clamp_int_val(
        emoji_in.get("max_per_short", defaults.captions.emoji.max_per_short),
        defaults.captions.emoji.max_per_short, 0, 5
    )
    style = _as_str(emoji_in.get("style", defaults.captions.emoji.style), defaults.captions.emoji.style)
    if style not in ("shiny", "pulse", "none"):
        style = defaults.captions.emoji.style
    emoji = EmojiConfig(enabled=enabled, max_per_short=max_per_short, style=style)

    # Strip punctuation flag (default True if missing)
    strip_punct = _as_bool(
        captions_in.get("strip_punctuation", defaults.captions.strip_punctuation),
        defaults.captions.strip_punctuation
    )

    # New sync/animation fields with safe defaults and ranges
    align_to_audio = _as_bool(
        captions_in.get("align_to_audio", defaults.captions.align_to_audio),
        defaults.captions.align_to_audio
    )
    fade_in_seconds = _clamp_float_val(
        captions_in.get("fade_in_seconds", defaults.captions.fade_in_seconds),
        defaults.captions.fade_in_seconds, 0.0, 5.0
    )
    fade_out_seconds = _clamp_float_val(
        captions_in.get("fade_out_seconds", defaults.captions.fade_out_seconds),
        defaults.captions.fade_out_seconds, 0.0, 5.0
    )

    captions = CaptionsConfig(
        font_size_px=fs,
        letter_spacing_px=ls,
        line_height=lh,
        base_color=base_color,
        shadow=shadow,
        accent_palette=accent_palette,
        animate=animate,
        position=position,
        emoji=emoji,
        strip_punctuation=strip_punct,
        align_to_audio=align_to_audio,
        fade_in_seconds=fade_in_seconds,
        fade_out_seconds=fade_out_seconds,
    )

    # Film Mode
    film_enabled = _as_bool(film_in.get("enabled", defaults.film_mode.enabled), defaults.film_mode.enabled)
    film_combo_duration = film_in.get("combo_duration", defaults.film_mode.combo_duration)
    film_single_duration = film_in.get("single_duration", defaults.film_mode.single_duration)
    film_max_moments = max(1, _as_int(film_in.get("max_moments", defaults.film_mode.max_moments), defaults.film_mode.max_moments))
    film_pause_threshold = _clamp(_as_float(film_in.get("pause_threshold", defaults.film_mode.pause_threshold), defaults.film_mode.pause_threshold), 0.1, 2.0)
    film_filler_words = film_in.get("filler_words", defaults.film_mode.filler_words)
    if not isinstance(film_filler_words, list):
        film_filler_words = defaults.film_mode.filler_words

    # Intelligent Pause Analysis
    intelligent_pause_in = film_in.get("intelligent_pause_analysis", {}) or {}
    if not isinstance(intelligent_pause_in, dict):
        intelligent_pause_in = {}

    intelligent_pause_enabled = _as_bool(
        intelligent_pause_in.get("enabled", defaults.film_mode.intelligent_pause_analysis.enabled),
        defaults.film_mode.intelligent_pause_analysis.enabled
    )
    intelligent_pause_model = _as_str(
        intelligent_pause_in.get("model", defaults.film_mode.intelligent_pause_analysis.model),
        defaults.film_mode.intelligent_pause_analysis.model
    )
    intelligent_pause_temperature = _clamp(
        _as_float(intelligent_pause_in.get("temperature", defaults.film_mode.intelligent_pause_analysis.temperature),
                  defaults.film_mode.intelligent_pause_analysis.temperature),
        0.0, 2.0
    )
    intelligent_pause_max_attempts = max(1, _as_int(
        intelligent_pause_in.get("max_attempts", defaults.film_mode.intelligent_pause_analysis.max_attempts),
        defaults.film_mode.intelligent_pause_analysis.max_attempts
    ))
    intelligent_pause_auto_trim_threshold = _clamp(
        _as_float(intelligent_pause_in.get("auto_trim_confidence_threshold",
                  defaults.film_mode.intelligent_pause_analysis.auto_trim_confidence_threshold),
                  defaults.film_mode.intelligent_pause_analysis.auto_trim_confidence_threshold),
        0.0, 1.0
    )
    intelligent_pause_batch_size = max(1, _as_int(
        intelligent_pause_in.get("batch_size", defaults.film_mode.intelligent_pause_analysis.batch_size),
        defaults.film_mode.intelligent_pause_analysis.batch_size
    ))
    intelligent_pause_cache_enabled = _as_bool(
        intelligent_pause_in.get("cache_enabled", defaults.film_mode.intelligent_pause_analysis.cache_enabled),
        defaults.film_mode.intelligent_pause_analysis.cache_enabled
    )
    intelligent_pause_cache_ttl = max(1, _as_int(
        intelligent_pause_in.get("cache_ttl_hours", defaults.film_mode.intelligent_pause_analysis.cache_ttl_hours),
        defaults.film_mode.intelligent_pause_analysis.cache_ttl_hours
    ))

    # Pause categories
    intelligent_pause_categories = intelligent_pause_in.get("pause_categories",
        defaults.film_mode.intelligent_pause_analysis.pause_categories)
    if not isinstance(intelligent_pause_categories, dict):
        intelligent_pause_categories = defaults.film_mode.intelligent_pause_analysis.pause_categories

    # Importance weights
    intelligent_pause_weights = intelligent_pause_in.get("importance_weights",
        defaults.film_mode.intelligent_pause_analysis.importance_weights)
    if not isinstance(intelligent_pause_weights, dict):
        intelligent_pause_weights = defaults.film_mode.intelligent_pause_analysis.importance_weights

    # API optimization
    intelligent_pause_api_opt = intelligent_pause_in.get("api_optimization",
        defaults.film_mode.intelligent_pause_analysis.api_optimization)
    if not isinstance(intelligent_pause_api_opt, dict):
        intelligent_pause_api_opt = defaults.film_mode.intelligent_pause_analysis.api_optimization

    intelligent_pause_analysis = IntelligentPauseAnalysisConfig(
        enabled=intelligent_pause_enabled,
        model=intelligent_pause_model,
        temperature=intelligent_pause_temperature,
        max_attempts=intelligent_pause_max_attempts,
        auto_trim_confidence_threshold=intelligent_pause_auto_trim_threshold,
        batch_size=intelligent_pause_batch_size,
        cache_enabled=intelligent_pause_cache_enabled,
        cache_ttl_hours=intelligent_pause_cache_ttl,
        pause_categories=intelligent_pause_categories,
        importance_weights=intelligent_pause_weights,
        api_optimization=intelligent_pause_api_opt
    )

    # Reference keywords for new ranking system
    film_reference_keywords = film_in.get("reference_keywords", defaults.film_mode.reference_keywords)
    if not isinstance(film_reference_keywords, dict):
        film_reference_keywords = defaults.film_mode.reference_keywords

    # Ranking weights
    film_ranking_weights = film_in.get("ranking_weights", defaults.film_mode.ranking_weights)
    if not isinstance(film_ranking_weights, dict):
        film_ranking_weights = defaults.film_mode.ranking_weights

    # LLM settings for film mode
    film_llm_model = _as_str(film_in.get("llm", {}).get("model", defaults.film_mode.llm_model), defaults.film_mode.llm_model)
    film_llm_temperature = _clamp(_as_float(film_in.get("llm", {}).get("temperature", defaults.film_mode.llm_temperature), defaults.film_mode.llm_temperature), 0.0, 2.0)
    film_llm_max_attempts = max(1, _as_int(film_in.get("llm", {}).get("max_attempts", defaults.film_mode.llm_max_attempts), defaults.film_mode.llm_max_attempts))
 
    # Ranking and fallback settings
    film_ranking_in = film_in.get("ranking", {}) or {}
    if not isinstance(film_ranking_in, dict):
        film_ranking_in = {}
    ranking_defaults = defaults.film_mode.ranking
    min_q = _clamp(
        _as_float(film_ranking_in.get("min_quality_threshold", ranking_defaults.get("min_quality_threshold")), ranking_defaults.get("min_quality_threshold")),
        0.0, 1.0
    )
    soft_min_q = _clamp(
        _as_float(film_ranking_in.get("soft_min_quality", ranking_defaults.get("soft_min_quality")), ranking_defaults.get("soft_min_quality")),
        0.0, 1.0
    )
    allow_fb = _as_bool(
        film_ranking_in.get("allow_fallback", ranking_defaults.get("allow_fallback")),
        ranking_defaults.get("allow_fallback")
    )
    fb_top_n = max(
        1,
        _as_int(film_ranking_in.get("fallback_top_n", ranking_defaults.get("fallback_top_n")), ranking_defaults.get("fallback_top_n"))
    )
    max_best = max(
        1,
        _as_int(film_ranking_in.get("max_best_moments", ranking_defaults.get("max_best_moments")), ranking_defaults.get("max_best_moments"))
    )
    film_ranking = {
        "min_quality_threshold": min_q,
        "soft_min_quality": soft_min_q,
        "allow_fallback": allow_fb,
        "fallback_top_n": fb_top_n,
        "max_best_moments": max_best,
    }
 
    # Дополнительные поля Film Mode v2 (с безопасными дефолтами и возможностью переопределения из YAML)
    film_window_minutes = max(1, _as_int(film_in.get("window_minutes", defaults.film_mode.window_minutes), defaults.film_mode.window_minutes))
    film_window_overlap_minutes = max(0, _as_int(film_in.get("window_overlap_minutes", defaults.film_mode.window_overlap_minutes), defaults.film_mode.window_overlap_minutes))
    film_max_mom_per_win = max(1, _as_int(film_in.get("max_moments_per_window", defaults.film_mode.max_moments_per_window), defaults.film_mode.max_moments_per_window))
    film_target_shorts = max(1, _as_int(film_in.get("target_shorts_count", defaults.film_mode.target_shorts_count), defaults.film_mode.target_shorts_count))
    film_generator_top_k = max(1, _as_int(film_in.get("generator_top_k", defaults.film_mode.generator_top_k), defaults.film_mode.generator_top_k))
    film_dedupe_iou = _clamp(_as_float(film_in.get("dedupe_iou_threshold", defaults.film_mode.dedupe_iou_threshold), defaults.film_mode.dedupe_iou_threshold), 0.0, 1.0)
    film_diversity_bucket = max(1, _as_int(film_in.get("diversity_bucket_minutes", defaults.film_mode.diversity_bucket_minutes), defaults.film_mode.diversity_bucket_minutes))
    film_min_combo_segments = max(1, _as_int(film_in.get("min_combo_segments", defaults.film_mode.min_combo_segments), defaults.film_mode.min_combo_segments))
    film_max_combo_segments = max(film_min_combo_segments, _as_int(film_in.get("max_combo_segments", defaults.film_mode.max_combo_segments), defaults.film_mode.max_combo_segments))

    film = FilmModeConfig(
        enabled=film_enabled,
        combo_duration=film_combo_duration,
        single_duration=film_single_duration,
        max_moments=film_max_moments,
        pause_threshold=film_pause_threshold,
        filler_words=film_filler_words,
        reference_keywords=film_reference_keywords,
        ranking_weights=film_ranking_weights,
        ranking=film_ranking,
        llm_model=film_llm_model,
        llm_temperature=film_llm_temperature,
        llm_max_attempts=film_llm_max_attempts,

        # v2 поля
        window_minutes=film_window_minutes,
        window_overlap_minutes=film_window_overlap_minutes,
        max_moments_per_window=film_max_mom_per_win,
        target_shorts_count=film_target_shorts,
        generator_top_k=film_generator_top_k,
        dedupe_iou_threshold=film_dedupe_iou,
        diversity_bucket_minutes=film_diversity_bucket,
        min_combo_segments=film_min_combo_segments,
        max_combo_segments=film_max_combo_segments,

        # Интеллектуальный анализ пауз
        intelligent_pause_analysis=intelligent_pause_analysis,
    )

    return AppConfig(processing=p, llm=l, logging=log, paths=paths, captions=captions, film_mode=film)


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


def reload_config() -> AppConfig:
    """
    Принудительно перезагружает конфигурацию из файла, сбрасывая кэш.
    Полезно при изменении config.yaml во время выполнения программы.
    """
    global _CONFIG
    _CONFIG = None  # Сбрасываем кэш
    return get_config()