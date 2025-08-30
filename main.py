from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extractAudio, crop_video, burn_captions, crop_bottom_video, animate_captions, get_video_dimensions
from Components.Transcription import transcribe_unified, prepare_words_for_segment
from faster_whisper import WhisperModel
import torch
import json
from Components.LanguageTasks import GetHighlights, build_transcription_prompt, compute_tone_and_keywords, compute_emojis_for_segment
from Components.FaceCrop import crop_to_70_percent_with_blur, crop_to_vertical_average_face
from Components.Database import VideoDatabase
from dataclasses import dataclass, field
from typing import Optional, List
import os
import traceback
import time
import math
from Components.config import get_config, AppConfig
from Components.Logger import logger, timed_operation
from Components.Paths import build_short_output_name

# Load config with reload to ensure latest changes are applied
from Components.config import reload_config
cfg = reload_config()
print(f"Конфиг загружен: shorts_dir={cfg.processing.shorts_dir}, model={cfg.llm.model_name}, crop_mode={cfg.processing.crop_mode}")

# Инициализация системы логирования
if cfg.logging.enable_system_info_logging:
    logger.log_system_info()

# --- Configuration Flags ---
# Set to True to use two words-level animated captions (slower but nicer)
# Set to False to use the default, faster ASS subtitle burning
USE_ANIMATED_CAPTIONS = cfg.processing.use_animated_captions

# Define the output directory for final shorts
SHORTS_DIR = cfg.processing.shorts_dir

# Define the crop percentage for the bottom of the video (useful when there are integrated captions in the original video)
CROP_PERCENTAGE_BOTTOM = cfg.processing.crop_bottom_percent

# --- Transcriptions JSON helpers (non-blocking) ---
def build_transcriptions_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent / "transcriptions"


def ensure_dir(path_like):
    from pathlib import Path
    p = Path(path_like)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def sanitize_base_name(name: str) -> str:
    import re
    from pathlib import Path
    try:
        base = Path(str(name)).stem
    except Exception:
        base = str(name)
    base = base.replace(" ", "_").lower()
    base = re.sub(r"[^A-Za-z0-9_-]", "", base)
    return base


def save_json_safely(data, path):
    import json
    from pathlib import Path
    p = Path(path)
    try:
        ensure_dir(p)
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        try:
            resolved = p.resolve()
        except Exception:
            resolved = str(p)
        print(f"[WARN] Ошибка при сохранении файла: {resolved} – {e}")
        return False


def _to_float(val, default=None):
    try:
        return float(val)
    except Exception:
        return default


def to_full_segments_payload(segments):
    result = []
    for seg in (segments or []):
        start = 0.0
        end = 0.0
        text = ""
        speaker = None
        confidence = None
        if isinstance(seg, dict):
            start = _to_float(seg.get("start"), 0.0)
            end = _to_float(seg.get("end"), 0.0)
            text = str(seg.get("text", ""))
            speaker_val = seg.get("speaker", None)
            speaker = str(speaker_val) if speaker_val is not None else None
            conf_val = seg.get("confidence", seg.get("prob", seg.get("probability", seg.get("score", None))))
            confidence = _to_float(conf_val, None) if conf_val is not None else None
        elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
            text = str(seg[0]) if seg[0] is not None else ""
            start = _to_float(seg[1], 0.0)
            end = _to_float(seg[2], 0.0)
            speaker = None
            confidence = None
        else:
            start = _to_float(getattr(seg, "start", 0.0), 0.0)
            end = _to_float(getattr(seg, "end", 0.0), 0.0)
            t = getattr(seg, "text", "")
            text = str(t) if t is not None else ""
            sp = getattr(seg, "speaker", None)
            speaker = str(sp) if sp is not None else None
            conf_val = getattr(seg, "confidence", None)
            if conf_val is None:
                for k in ("prob", "probability", "score"):
                    if hasattr(seg, k):
                        conf_val = getattr(seg, k)
                        break
            confidence = _to_float(conf_val, None) if conf_val is not None else None
        result.append({"start": start, "end": end, "text": text, "speaker": speaker, "confidence": confidence})
    return {"segments": result}


def to_words_payload(word_level_result):
    words = []
    segments = []
    if word_level_result is None:
        return {"words": words}
    if isinstance(word_level_result, dict):
        segments = word_level_result.get("segments", [])
        if not segments and "words" in word_level_result:
            segments = [{"words": word_level_result.get("words", [])}]
    else:
        segments = getattr(word_level_result, "segments", []) or []
    for seg in segments:
        seg_words = seg.get("words", []) if isinstance(seg, dict) else getattr(seg, "words", []) or []
        # Wait: indentation alignment; ensure 8 spaces? We'll keep 8 spaces. We'll correct.
        for w in seg_words:
            if isinstance(w, dict):
                start = _to_float(w.get("start", 0.0), 0.0)
                end = _to_float(w.get("end", 0.0), 0.0)
                word_val = w.get("word", w.get("text", ""))
                conf_val = w.get("confidence", w.get("prob", w.get("probability", w.get("score", None))))
            else:
                start = _to_float(getattr(w, "start", 0.0), 0.0)
                end = _to_float(getattr(w, "end", 0.0), 0.0)
                word_val = getattr(w, "word", getattr(w, "text", ""))
                conf_val = getattr(w, "confidence", None)
                if conf_val is None:
                    for k in ("prob", "probability", "score"):
                        if hasattr(w, k):
                            conf_val = getattr(w, k)
                            break
            words.append({
                "start": start,
                "end": end,
                "word": str(word_val) if word_val is not None else "",
                "confidence": _to_float(conf_val, None) if conf_val is not None else None
            })
    return {"words": words}

@dataclass
class ProcessingContext:
    """
    Контекст обработки видео для пайплайна.

    Ключевые поля:
    - cfg: AppConfig
    - db: VideoDatabase
    - url/local_path: входной источник
    - video_path: путь к исходному видео
    - video_id: идентификатор в БД
    - audio_path: путь к извлеченному аудио
    - base_name: базовое имя файла видео
    - initial_width/initial_height: исходные размеры видео
    - transcription_segments: список сегментов транскрипции (dict: start, end, text, speaker?)
    - transcription_text: форматированный текст транскрипции для LLM
    - outputs: список путей к финальным клипам
    """
    cfg: AppConfig
    db: VideoDatabase
    url: Optional[str] = None
    local_path: Optional[str] = None
    video_path: Optional[str] = None
    video_id: Optional[str] = None
    audio_path: Optional[str] = None
    base_name: Optional[str] = None
    initial_width: int = 0
    initial_height: int = 0
    transcription_segments: List[dict] = field(default_factory=list)
    transcription_text: Optional[str] = None
    outputs: List[str] = field(default_factory=list)
    word_level_transcription: Optional[dict] = None


def init_context(url: Optional[str], local_path: Optional[str]) -> ProcessingContext:
    """Инициализирует контекст: загружает конфиг, создаёт БД и гарантирует наличие выходных директорий."""
    cfg_local = get_config()
    # Ensure directories exist
    os.makedirs(cfg_local.processing.shorts_dir, exist_ok=True)
    os.makedirs(cfg_local.processing.videos_dir, exist_ok=True)
    ctx = ProcessingContext(cfg=cfg_local, db=VideoDatabase(), url=url, local_path=local_path)
    return ctx


@timed_operation("resolve_video_source")
def resolve_video_source(ctx: ProcessingContext) -> bool:
    """Определяет источник видео (URL/локальный), учитывает кэш БД, выставляет ctx.video_path и ctx.video_id."""
    if not ctx.url and not ctx.local_path:
        logger.logger.error("Error: Must provide either URL or local path")
        return False

    video_path = None
    video_id = None

    if ctx.url:
        logger.logger.info(f"Processing YouTube URL: {ctx.url}")
        cached_data = ctx.db.get_cached_processing(youtube_url=ctx.url)
        if cached_data:
            logger.logger.info("Found cached video from URL!")
            video_path = cached_data["video"][2]
            video_id = cached_data["video"][0]
            if not os.path.exists(video_path):
                logger.logger.warning(f"Cached video path not found: {video_path}. Re-downloading.")
                video_path = None
                video_id = None
        if not video_path:
            with logger.operation_context("download_youtube_video", {"url": ctx.url}):
                video_path = download_youtube_video(ctx.url)
                if not video_path:
                    logger.logger.error("Failed to download video")
                    return False
                if not video_path.lower().endswith('.mp4'):
                    base, _ = os.path.splitext(video_path)
                    new_path = base + ".mp4"
                    try:
                        os.rename(video_path, new_path)
                        video_path = new_path
                        logger.logger.info(f"Renamed downloaded file to: {video_path}")
                    except OSError as e:
                        logger.logger.warning(f"Error renaming file to mp4: {e}. Trying conversion.")
                        pass
    else:
        logger.logger.info(f"Processing local file: {ctx.local_path}")
        if not os.path.exists(ctx.local_path):
            logger.logger.error("Error: Local file does not exist")
            return False
        video_path = ctx.local_path
        cached_data = ctx.db.get_cached_processing(local_path=ctx.local_path)
        if cached_data:
            logger.logger.info("Found cached local video!")
            video_id = cached_data["video"][0]

    if not video_path or not os.path.exists(video_path):
        logger.logger.error("No valid video path obtained or file does not exist.")
        return False

    ctx.video_path = video_path
    ctx.video_id = video_id
    ctx.base_name = os.path.splitext(os.path.basename(video_path))[0]
    return True


def validate_dimensions(ctx: ProcessingContext) -> bool:
    """Проверяет исходные размеры видео и сохраняет их в контекст."""
    print("\n--- Checking Initial Video Dimensions ---")
    w, h = get_video_dimensions(ctx.video_path)
    if w is None or h is None:
        print("Error: Could not determine initial video dimensions. Aborting.")
        return False
    ctx.initial_width, ctx.initial_height = int(w), int(h)
    if ctx.initial_width < ctx.cfg.processing.min_video_dimension_px or ctx.initial_height < ctx.cfg.processing.min_video_dimension_px:
        print(f"Warning: Initial video dimensions ({ctx.initial_width}x{ctx.initial_height}) seem very small.")
    print("--- Initial Check Done ---")
    return True


def ensure_audio(ctx: ProcessingContext) -> bool:
    """Извлекает или загружает из кэша аудио. Обновляет БД и ctx.audio_path/ctx.video_id."""
    audio_path = None
    cached_data = None
    if ctx.url:
        cached_data = ctx.db.get_cached_processing(youtube_url=ctx.url)
    else:
        cached_data = ctx.db.get_cached_processing(local_path=ctx.local_path or ctx.video_path)

    if cached_data and cached_data["video"][3]:
        print("Using cached audio file reference")
        audio_path = cached_data["video"][3]
        if not os.path.exists(audio_path):
            print("Cached audio file not found, extracting again")
            audio_path = None

    if not audio_path:
        print(f"Extracting audio from video: {ctx.video_path}")
        audio_path = extractAudio(ctx.video_path)
        if not audio_path:
            print("Failed to extract audio")
            return False
        if ctx.video_id:
            ctx.db.update_video_audio_path(ctx.video_id, audio_path)
        else:
            ctx.video_id = ctx.db.add_video(ctx.url, ctx.video_path, audio_path)

    if not ctx.video_id:
        video_entry = ctx.db.get_video(youtube_url=ctx.url, local_path=ctx.video_path)
        if video_entry:
            ctx.video_id = video_entry[0]
        else:
            print("Error: Video ID could not be determined after processing.")
            return False

    ctx.audio_path = audio_path
    return True


def run_unified_transcription(ctx: ProcessingContext, model: WhisperModel) -> bool:
    """
    Выполняет или загружает из кэша транскрипцию (сегменты и слова).
    Использует JSON-файлы как основной кэш для обоих типов данных.
    Сохраняет результаты в контекст.
    """
    base_name = sanitize_base_name(os.path.splitext(os.path.basename(ctx.video_path))[0])
    segments_cache_path = build_transcriptions_dir() / f"{base_name}_full_segments.json"
    words_cache_path = build_transcriptions_dir() / f"{base_name}_word_level.json"

    segments_loaded = False
    words_loaded = False

    # Попытка загрузить из JSON кэша
    if segments_cache_path.exists():
        try:
            with segments_cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                ctx.transcription_segments = data.get("segments", [])
                segments_loaded = True
                print("Loaded segment transcription from JSON cache.")
        except Exception as e:
            print(f"[WARN] Could not load segments from JSON cache: {e}")

    if words_cache_path.exists():
        try:
            with words_cache_path.open("r", encoding="utf-8") as f:
                ctx.word_level_transcription = json.load(f)
                words_loaded = True
                print("Loaded word-level transcription from JSON cache.")
        except Exception as e:
            print(f"[WARN] Could not load words from JSON cache: {e}")

    if segments_loaded and words_loaded:
        print("Both transcriptions loaded from cache. Skipping transcription.")
        # Убедимся, что в контексте не None, а пустой dict, если что-то пошло не так
        if ctx.word_level_transcription is None:
            ctx.word_level_transcription = {"segments": []}
        return True

    # Если чего-то нет, запускаем унифицированную транскрипцию
    print("Cache incomplete. Running unified transcription for segments and words...")
    
    # Используется унифицированный подход для повышения эффективности
    segments_legacy, word_level_transcription = transcribe_unified(ctx.audio_path, model)

    if not segments_legacy:
        print("Unified transcription failed. Cannot proceed.")
        return False

    # Преобразуем и сохраняем результаты в контекст
    full_segments_payload = to_full_segments_payload(segments_legacy)
    ctx.transcription_segments = full_segments_payload.get("segments", [])
    ctx.word_level_transcription = word_level_transcription

    # Сохраняем в БД (как и раньше) и в JSON-кэш
    if ctx.video_id:
        ctx.db.add_transcription(ctx.video_id, segments_legacy)
    
    save_json_safely(full_segments_payload, segments_cache_path)
    save_json_safely(word_level_transcription, words_cache_path)
    
    print("Unified transcription complete. Results saved to context and cache.")
    return True


def prepare_transcript_text(ctx: ProcessingContext) -> None:
    """Формирует текст транскрипции через LanguageTasks.build_transcription_prompt и логирует превью."""
    TransText = build_transcription_prompt(ctx.transcription_segments)
    ctx.transcription_text = TransText
    print(f"\nFirst {cfg.processing.log_transcription_preview_len} characters of transcription:")
    print(TransText[:cfg.processing.log_transcription_preview_len] + "...")


def fetch_highlights(ctx: ProcessingContext) -> list:
    """Запрашивает у LLM список обогащённых хайлайтов по готовому тексту транскрипции."""
    print("Generating new highlights")
    return GetHighlights(ctx.transcription_text or "")


@timed_operation("process_highlight")
def process_highlight(ctx: ProcessingContext, item) -> Optional[str]:
    """Обрабатывает один хайлайт: кропы, субтитры, сохранение; возвращает путь финального файла либо None."""
    temp_segment = None
    cropped_vertical_temp = None
    cropped_vertical_final = None
    segment_audio_path = None
    transcription_result = None

    seq = int(item.get("_seq", 1)) if isinstance(item, dict) else 1
    total = int(item.get("_total", 1)) if isinstance(item, dict) else 1

    try:
        start = float(item["start"])
        stop = float(item["end"])

        # Корректировка stop по фактическому окончанию последнего полного слова (по start < stop)
        adjusted_stop = stop
        if getattr(ctx, "word_level_transcription", None):
            last_word_end = find_last_word_end_time(ctx.word_level_transcription, stop)
            if last_word_end is not None and last_word_end > stop:
                prev_stop = adjusted_stop
                adjusted_stop = last_word_end
                logger.logger.info(f"[WordLevel] Adjusted stop from {prev_stop:.2f}s to {adjusted_stop:.2f}s based on last word end")
        # Гарантия: stop > start
        if adjusted_stop <= start:
            adjusted_stop = start + 0.1

        # Округление времени окончания short в положительную сторону для полного слова
        prev_adjusted_stop = adjusted_stop
        adjusted_stop = math.ceil(adjusted_stop)
        if adjusted_stop != prev_adjusted_stop:
            logger.logger.info(f"[Ceil] Rounded up stop from {prev_adjusted_stop:.2f}s to {adjusted_stop:.2f}s")

        logger.logger.info(f"\n--- Processing Highlight {seq}/{total}: Start={start:.2f}s, End={stop:.2f}s (effective end {adjusted_stop:.2f}s) ---")
        if isinstance(item, dict) and "caption_with_hashtags" in item:
            logger.logger.info(f"Caption: {item['caption_with_hashtags']}")

        # --- Define File Paths ---
        base_name = os.path.splitext(os.path.basename(ctx.video_path))[0]
        output_base = f"{base_name}_highlight_{seq}"
        temp_segment = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_temp_segment.mp4")
        cropped_vertical_temp = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_vertical_temp.mp4")
        cropped_vertical_final = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_vertical_final.mp4")
        # Use unified naming with zero-padded index for final output (and derived temp anim path)
        final_output_with_captions, _unused_temp_anim = build_short_output_name(base_name, seq, SHORTS_DIR)
        if USE_ANIMATED_CAPTIONS:
            segment_audio_path = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_temp_audio.wav")

        # 1. Extract Segment (Video + Audio, Original Aspect Ratio)
        with logger.operation_context("extract_segment", {"start": start, "end": adjusted_stop}):
            logger.logger.info("1. Extracting segment...")
            extract_success = crop_video(ctx.video_path, temp_segment, start, adjusted_stop, ctx.initial_width, ctx.initial_height)
            if not extract_success:
                logger.logger.error(f"Failed step 1 for highlight {seq}. Skipping.")
                if os.path.exists(temp_segment):
                    try:
                        os.remove(temp_segment)
                    except Exception as clean_e:
                        logger.logger.warning(f"Warning: Could not remove temp segment file: {clean_e}")
                return None

        # --- CHECK DIMENSIONS: Segment ---
        with logger.operation_context("check_segment_dimensions", {"segment_path": temp_segment}):
            logger.logger.info("\n--- Checking Segment Video Dimensions ---")
            segment_width, segment_height = get_video_dimensions(temp_segment)
            if segment_width is None or segment_height is None:
                logger.logger.error("Error: Could not determine segment video dimensions. Skipping highlight.")
                if os.path.exists(temp_segment):
                    try:
                        os.remove(temp_segment)
                    except Exception as clean_e:
                        logger.logger.warning(f"Warning: Could not remove temp segment file: {clean_e}")
                return None
            if segment_width != ctx.initial_width or segment_height != ctx.initial_height:
                logger.logger.warning(f"Warning: Segment dimensions ({segment_width}x{segment_height}) differ from initial ({ctx.initial_width}x{ctx.initial_height}).")
            logger.logger.info("--- Segment Check Done ---")

        # 2. Create Vertical Crop (Based on crop_mode configuration)
        with logger.operation_context("create_vertical_crop", {"segment_path": temp_segment}):
            crop_mode = cfg.processing.crop_mode
            if crop_mode == "70_percent_blur":
                logger.logger.info("2. Creating 70% width crop with blur background...")
                crop_function = crop_to_70_percent_with_blur
                crop_error_msg = "70% crop with blur"
            elif crop_mode == "average_face":
                logger.logger.info("2. Creating average face centered vertical crop...")
                crop_function = crop_to_vertical_average_face
                crop_error_msg = "average face crop"
            else:
                logger.logger.warning(f"Unknown crop_mode '{crop_mode}', falling back to 70_percent_blur")
                crop_function = crop_to_70_percent_with_blur
                crop_error_msg = "70% crop with blur (fallback)"

            vert_crop_path = crop_function(temp_segment, cropped_vertical_temp)
            if not vert_crop_path:
                logger.logger.error(f"Failed step 2 ({crop_error_msg}) for highlight {seq}. Skipping.")
                if os.path.exists(temp_segment):
                    try:
                        os.remove(temp_segment)
                    except Exception as clean_e:
                        logger.logger.warning(f"Warning: Could not remove temp segment file: {clean_e}")
                if os.path.exists(cropped_vertical_temp):
                    try:
                        os.remove(cropped_vertical_temp)
                    except Exception as clean_e:
                        logger.logger.warning(f"Warning: Could not remove temp vertical crop file: {clean_e}")
                return None
            cropped_vertical_temp = vert_crop_path

        # 3. Crop Bottom Off Vertical Video (Temporary Fix)
        if CROP_PERCENTAGE_BOTTOM > 0:
            print("3. Applying bottom crop to vertical video...")
            bottom_crop_success = crop_bottom_video(cropped_vertical_temp, cropped_vertical_final, CROP_PERCENTAGE_BOTTOM)
            if not bottom_crop_success:
                print(f"Failed step 3 for highlight {seq}. Skipping.")
                if os.path.exists(temp_segment):
                    try:
                        os.remove(temp_segment)
                    except Exception:
                        pass
                if os.path.exists(cropped_vertical_temp):
                    try:
                        os.remove(cropped_vertical_temp)
                    except Exception:
                        pass
                if os.path.exists(cropped_vertical_final):
                    try:
                        os.remove(cropped_vertical_final)
                    except Exception:
                        pass
                return None
        else:
            print("No bottom crop applied")
            cropped_vertical_final = cropped_vertical_temp

        # 4. Choose Captioning Method
        captioning_success = False
        if USE_ANIMATED_CAPTIONS:
            print("Attempting Word-Level Animated Captions (reusing global word-level transcription)...")
            transcription_result = None
            if getattr(ctx, "word_level_transcription", None):
                # Подготовка слов через чистый хелпер
                try:
                    transcription_result = prepare_words_for_segment(ctx.word_level_transcription, start, adjusted_stop)
                    # Логирование количества слов после фильтрации
                    words_count = 0
                    try:
                        segs = transcription_result.get("segments", []) or []
                        if segs:
                            words_count = len(segs[0].get("words", []))
                    except Exception:
                        words_count = 0
                    print(f"[WordLevel] Prepared {words_count} words for animated captions from global transcription.")
                    # Опционально сохраним JSON слов для дебага/просмотра
                    try:
                        base_sanitized = sanitize_base_name(os.path.splitext(os.path.basename(ctx.video_path))[0])
                        words_payload = to_words_payload(transcription_result)
                        target_words = build_transcriptions_dir() / f"{base_sanitized}_highlight_{seq}_words.json"
                        save_json_safely(words_payload, target_words)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[WordLevel][WARN] Failed to prepare words subset: {e}")
                    transcription_result = None
            else:
                print("[WordLevel][WARN] No global word-level transcription available; cannot animate captions for this highlight.")
    
            if transcription_result and transcription_result.get("segments", []) and transcription_result["segments"][0].get("words"):
                # tone/keywords heuristic — compute meta based on segment text used for captions
                text_for_segment = ""
                try:
                    # Prefer enriched text from highlight item (LLM-extracted for this segment)
                    text_for_segment = (item.get("segment_text", "") if isinstance(item, dict) else "") or ""
                except Exception:
                    text_for_segment = ""
                if not text_for_segment:
                    # Fallback: reconstruct from prepared words list
                    try:
                        segs = transcription_result.get("segments", []) or []
                        if segs:
                            words = segs[0].get("words", []) or []
                            text_for_segment = " ".join(
                                (w.get("text") or w.get("word") or "").strip()
                                for w in words if isinstance(w, dict) and (w.get("text") or w.get("word"))
                            ).strip()
                    except Exception:
                        text_for_segment = ""
                meta = compute_tone_and_keywords(text_for_segment) if text_for_segment else None

                # emoji: heuristics and placement — propagate emoji metadata (backward compatible)
                highlight_meta = meta or {}
                try:
                    cfg_emoji = getattr(ctx.cfg.captions, "emoji", None)
                    if cfg_emoji and getattr(cfg_emoji, "enabled", False) and text_for_segment:
                        tone_val = (highlight_meta.get("tone") if isinstance(highlight_meta, dict) else None) or "neutral"
                        max_per = int(getattr(cfg_emoji, "max_per_short", 0) or 0)
                        emojis = compute_emojis_for_segment(text_for_segment, tone_val, max_per)
                        if isinstance(highlight_meta, dict):
                            highlight_meta = {**highlight_meta, "emojis": list(emojis or [])[:max_per]}
                except Exception:
                    # Полная обратная совместимость: любые ошибки с эмодзи не должны ломать рендер
                    pass

                captioning_success = animate_captions(
                    cropped_vertical_final,
                    temp_segment,
                    transcription_result,
                    final_output_with_captions,
                    style_cfg=ctx.cfg.captions,
                    highlight_meta=highlight_meta
                )
            else:
                print("Word-level data for this segment is empty. Skipping animation.")
                captioning_success = False
        else:
            print("Using Standard ASS Caption Burning...")
            transcriptions_legacy = [[
                str(seg.get("text", "")),
                float(seg.get("start", 0.0)),
                float(seg.get("end", 0.0)),
            ] for seg in (ctx.transcription_segments or [])]
            captioning_success = burn_captions(cropped_vertical_final, temp_segment, transcriptions_legacy, start, adjusted_stop, final_output_with_captions, style_cfg=ctx.cfg.captions)

        # 5. Handle Captioning Result
        if not captioning_success:
            print(f"Animated caption generation failed for highlight {seq}. Attempting ASS burn fallback...")
            transcriptions_legacy = [[
                str(seg.get("text", "")),
                float(seg.get("start", 0.0)),
                float(seg.get("end", 0.0)),
            ] for seg in (ctx.transcription_segments or [])]
            fallback_success = burn_captions(cropped_vertical_final, temp_segment, transcriptions_legacy, start, adjusted_stop, final_output_with_captions, style_cfg=ctx.cfg.captions)
            if not fallback_success:
                print(f"ASS fallback failed for highlight {seq}. Skipping.")
                if os.path.exists(temp_segment):
                    try:
                        os.remove(temp_segment)
                    except Exception as clean_e:
                        print(f"Warning: Could not remove temp segment file: {clean_e}")
                if os.path.exists(cropped_vertical_temp):
                    try:
                        os.remove(cropped_vertical_temp)
                    except Exception as clean_e:
                        print(f"Warning: Could not remove temp vertical file: {clean_e}")
                if os.path.exists(cropped_vertical_final):
                    try:
                        os.remove(cropped_vertical_final)
                    except Exception as clean_e:
                        print(f"Warning: Could not remove final vertical file: {clean_e}")
                if segment_audio_path and os.path.exists(segment_audio_path):
                    try:
                        os.remove(segment_audio_path)
                    except Exception as clean_e:
                        print(f"Warning: Could not remove segment audio file: {clean_e}")
                return None
            else:
                captioning_success = True

        logger.logger.info(f"Successfully processed highlight {seq}.")
        ctx.outputs.append(final_output_with_captions)
        logger.logger.info(f"Saving highlight {seq} info to database: {final_output_with_captions}")

        segment_text = item.get('segment_text', '') if isinstance(item, dict) else ''
        caption = item.get('caption_with_hashtags', '') if isinstance(item, dict) else ''

        with logger.operation_context("save_to_database", {"video_id": ctx.video_id, "highlight_path": final_output_with_captions}):
            ctx.db.add_highlight(
                ctx.video_id,
                start,
                adjusted_stop,
                final_output_with_captions,
                segment_text=segment_text,
                caption_with_hashtags=caption
            )

        # --- Cleanup Intermediate Files ---
        with logger.operation_context("cleanup_intermediate_files", {"highlight_seq": seq}):
            logger.logger.info("Cleaning up intermediate files for this highlight...")
            if os.path.exists(temp_segment):
                try:
                    os.remove(temp_segment)
                except Exception as clean_e:
                    logger.logger.warning(f"Warning: Could not remove temp segment file: {clean_e}")
            if os.path.exists(cropped_vertical_temp):
                try:
                    os.remove(cropped_vertical_temp)
                except Exception as clean_e:
                    logger.logger.warning(f"Warning: Could not remove temp vertical file: {clean_e}")
            if os.path.exists(cropped_vertical_final):
                try:
                    os.remove(cropped_vertical_final)
                except Exception as clean_e:
                    logger.logger.warning(f"Warning: Could not remove final vertical file: {clean_e}")
            if segment_audio_path and os.path.exists(segment_audio_path):
                try:
                    os.remove(segment_audio_path)
                except Exception as clean_e:
                    logger.logger.warning(f"Warning: Could not remove segment audio file: {clean_e}")

        return final_output_with_captions

    except Exception:
        print(f"\n--- Error processing highlight {seq} --- ")
        traceback.print_exc()
        print("Continuing to next highlight if available.")
        if temp_segment and os.path.exists(temp_segment):
            try:
                os.remove(temp_segment)
            except Exception as clean_e:
                print(f"Warning: Could not remove temp segment file: {clean_e}")
        if cropped_vertical_temp and os.path.exists(cropped_vertical_temp):
            try:
                os.remove(cropped_vertical_temp)
            except Exception as clean_e:
                print(f"Warning: Could not remove temp vertical file: {clean_e}")
        if cropped_vertical_final and os.path.exists(cropped_vertical_final):
            try:
                os.remove(cropped_vertical_final)
            except Exception as clean_e:
                print(f"Warning: Could not remove final vertical file: {clean_e}")
        if segment_audio_path and os.path.exists(segment_audio_path):
            try:
                os.remove(segment_audio_path)
            except Exception as clean_e:
                print(f"Warning: Could not remove segment audio file: {clean_e}")
        return None


def process_all_highlights(ctx: ProcessingContext, items: list) -> List[str]:
    """Итерирует по сегментам, вызывает process_highlight, накапливает выходы и печатает итоги/ошибки."""
    try:
        final_output_paths: List[str] = []
        total = len(items or [])
        for i, raw_item in enumerate(items or []):
            item = dict(raw_item) if isinstance(raw_item, dict) else raw_item
            if isinstance(item, dict):
                item["_seq"] = i + 1
                item["_total"] = total
            out_path = process_highlight(ctx, item)
            if out_path:
                final_output_paths.append(out_path)

        if not final_output_paths:
            print("\nProcessing finished, but no highlight segments were successfully converted.")
            return []
        else:
            print(f"\nProcessing finished. Generated {len(final_output_paths)} shorts in '{SHORTS_DIR}' directory.")
            return final_output_paths
    except Exception as e:
        print(f"Error in overall highlight processing: {str(e)}")
        traceback.print_exc()
        return []


def _select_whisper_runtime():
    """Выбирает оптимальные параметры для модели Whisper с GPU-first подходом."""
    try:
        has_cuda = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if has_cuda else 0
    except Exception:
        has_cuda = False
        gpu_count = 0

    # GPU-first подход
    if has_cuda and cfg.logging.gpu_priority_mode:
        device = "cuda"
        # Используем лучшую доступную модель для GPU
        model_size = "large-v3"
        compute_type = "float16"
        cpu_threads = 0  # Для GPU используем 0 CPU threads
        logger.logger.info(f"GPU-first режим: Используем {gpu_count} GPU(s), модель {model_size}")
    else:
        device = "cpu"
        model_size = "small"
        compute_type = "int8"
        cpu_threads = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
        logger.logger.info(f"CPU режим: Используем {cpu_threads} потоков, модель {model_size}")

    return model_size, device, compute_type, cpu_threads


@timed_operation("video_processing_pipeline")
def process_video(url: str = None, local_path: str = None):
    """
    Координатор пайплайна. Публичная сигнатура сохранена.
    Возвращает список путей к сгенерированным клипам или None при ошибке/пустом результате.
    """
    with logger.operation_context("initialize_context", {"url": url, "local_path": local_path}):
        ctx = init_context(url, local_path)

    with logger.operation_context("resolve_video_source", {"url": url, "local_path": local_path}):
        if not resolve_video_source(ctx):
            return None

    with logger.operation_context("validate_dimensions", {"video_path": ctx.video_path}):
        if not validate_dimensions(ctx):
            return None

    with logger.operation_context("ensure_audio", {"video_path": ctx.video_path}):
        if not ensure_audio(ctx):
            return None

    # --- Загрузка модели Whisper ---
    with logger.operation_context("load_whisper_model", {"model_size": "auto", "device": "auto"}):
        print("\n--- Loading Whisper Model ---")
        model_size, device, compute_type, cpu_threads = _select_whisper_runtime()
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        try:
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads if device == "cpu" else 0,
                num_workers=2,
            )
            print(f"Faster-Whisper model loaded: {model_size} on {device} ({compute_type})")
        except Exception as e:
            print(f"FATAL: Could not load Whisper model. Error: {e}")
            return None

    # --- Транскрипция (унифицированный вызов) ---
    with logger.operation_context("transcription", {"audio_path": ctx.audio_path}):
        logger.logger.info(f"Начало транскрипции аудио: {ctx.audio_path}")
        transcription_start = time.time()

        if not run_unified_transcription(ctx, model):
            logger.logger.error("Транскрипция завершилась неудачей")
            return None

        transcription_time = time.time() - transcription_start
        logger.logger.info(f"Транскрипция завершена за {transcription_time:.2f} секунд")

    with logger.operation_context("prepare_transcript_text", {"transcription_length": len(ctx.transcription_text or "")}):
        prepare_transcript_text(ctx)

    try:
        with logger.operation_context("fetch_highlights", {"transcription_length": len(ctx.transcription_text or "")}):
            highlights = fetch_highlights(ctx)
            if not highlights or len(highlights) == 0:
                print("No valid highlights found")
                return None

        with logger.operation_context("process_highlights", {"highlights_count": len(highlights)}):
            # Создаем прогресс-бар для обработки хайлайтов
            progress_bar = logger.create_progress_bar(
                total=len(highlights),
                desc="Обработка хайлайтов",
                unit="highlight"
            )

            outputs = []
            for i, highlight in enumerate(highlights):
                # Ensure proper sequencing for unique filenames and logging
                payload = dict(highlight) if isinstance(highlight, dict) else highlight
                if isinstance(payload, dict):
                    payload["_seq"] = i + 1
                    payload["_total"] = len(highlights)
                with logger.operation_context(
                    "process_single_highlight",
                    {"highlight_index": i, "highlight_text": (payload.get("caption_with_hashtags", "")[:50] if isinstance(payload, dict) else "")}
                ):
                    output = process_highlight(ctx, payload)
                    if output:
                        outputs.append(output)

                progress_bar.update(1)
                progress_bar.set_postfix({
                    "Обработано": f"{i+1}/{len(highlights)}",
                    "Успешно": len(outputs)
                })

            progress_bar.close()

        if not outputs:
            return None
        return outputs
    except Exception as e:
        print(f"Error in overall highlight processing: {str(e)}")
        traceback.print_exc()
        return None


    

def find_last_word_end_time(word_level_transcription: dict, segment_end_time: float) -> Optional[float]:
    """
    Определяет фактическое время окончания «последнего слова до segment_end_time».

    Определение «последнего слова до segment_end_time»:
    - Рассматриваются только слова, у которых start < segment_end_time (start — абсолютный).
    - Возвращается максимальный end среди таких слов.

    Почему функция может вернуть end > segment_end_time:
    - Слово может начинаться до segment_end_time, а заканчиваться ПОСЛЕ него. Это важно для
      корректировки stop вправо, чтобы не обрывать слово в анимации (используется max(original_stop, last_word_end)).

    Граничные случаи:
    - Отсутствуют сегменты/слова — возвращается None.
    - Нечисловые или отсутствующие start/end у слова — такие слова пропускаются.
    - Порядок слов/сегментов может быть произвольным — берётся максимум end среди подходящих.

    Безопасность:
    - Функция устойчиво обрабатывает некорректные структуры (не dict / пустые поля), возвращая None.
    """
    try:
        if not isinstance(word_level_transcription, dict):
            return None

        segments = word_level_transcription.get("segments", []) or []
        if not isinstance(segments, list) or not segments:
            return None

        last_end: Optional[float] = None
        for seg in segments:
            # Безопасно достаём список слов
            words = seg.get("words", []) if isinstance(seg, dict) else getattr(seg, "words", []) or []
            if not words:
                continue

            for w in words:
                if isinstance(w, dict):
                    s = _to_float(w.get("start", None), None)
                    e = _to_float(w.get("end", None), None)
                else:
                    s = _to_float(getattr(w, "start", None), None)
                    e = _to_float(getattr(w, "end", None), None)

                # Пропускаем слова без числовых таймкодов
                if s is None or e is None:
                    continue

                # Критерий отбора — слово началось до segment_end_time
                if s < segment_end_time:
                    if last_end is None or e > last_end:
                        last_end = e

        return last_end
    except Exception:
        return None


if __name__ == "__main__":
    print("\nVideo Processing Options:")
    print("1. Process YouTube URL")
    print("2. Process Local File")
    print("3. Process Film Mode (analyze best moments)")
    choice = input("Enter your choice (1, 2, or 3): ")

    if choice == "1":
        url = input("Enter YouTube URL: ")
        output = process_video(url=url)
    elif choice == "2":
        local_file = input("Enter path to local video file: ")
        output = process_video(local_path=local_file)
    elif choice == "3":
        print("\n🎬 Режим 'Фильм' - Анализ лучших моментов")
        print("Выберите источник видео:")

        # Импортируем функции для работы с папкой movies
        from Components.FilmMode import scan_movies_folder, display_movie_selection, select_movie_by_number, analyze_film_main

        # Сканируем папку movies
        video_files = scan_movies_folder()

        # Показываем список файлов
        display_movie_selection(video_files)

        # Если есть файлы, предлагаем выбрать
        if video_files:
            selected_file = select_movie_by_number(video_files)

            if selected_file == "URL_INPUT":
                # Пользователь выбрал ручной ввод
                print("\nВыберите тип источника:")
                print("1. YouTube URL")
                print("2. Путь к локальному файлу")
                manual_choice = input("Введите ваш выбор (1 или 2): ").strip()

                if manual_choice == "1":
                    url = input("Введите YouTube URL: ").strip()
                    result = analyze_film_main(url=url)
                elif manual_choice == "2":
                    local_file = input("Введите путь к видео файлу: ").strip()
                    if not os.path.exists(local_file):
                        print(f"❌ Ошибка: Файл не найден: {local_file}")
                        result = None
                    else:
                        result = analyze_film_main(local_path=local_file)
                else:
                    print("❌ Неверный выбор")
                    result = None
            elif selected_file:
                # Выбран файл из папки movies
                # Дополнительная проверка существования файла
                if not os.path.exists(selected_file):
                    print(f"❌ Ошибка: Файл не найден: {selected_file}")
                    print("Файл мог быть удален после сканирования.")
                    result = None
                else:
                    result = analyze_film_main(local_path=selected_file)
            else:
                # Отмена выбора
                print("Выбор отменен.")
                result = None
        else:
            # Папка movies пуста, предлагаем ручной ввод
            print("\nПапка movies пуста. Выберите тип источника:")
            print("1. YouTube URL")
            print("2. Путь к локальному файлу")
            manual_choice = input("Введите ваш выбор (1 или 2): ").strip()

            if manual_choice == "1":
                url = input("Введите YouTube URL: ").strip()
                result = analyze_film_main(url=url)
            elif manual_choice == "2":
                local_file = input("Введите путь к видео файлу: ").strip()
                if not os.path.exists(local_file):
                    print(f"❌ Ошибка: Файл не найден: {local_file}")
                    result = None
                else:
                    result = analyze_film_main(local_path=local_file)
            else:
                print("❌ Неверный выбор")
                result = None

        if result:
            print(f"\nFilm analysis completed successfully!")
            print(f"Video ID: {result.video_id}")
            print(f"Duration: {result.duration:.1f} seconds")
            print(f"Found {len(result.keep_ranges)} best moments")
            print(f"Generated {len(result.generated_shorts)} shorts")
            print(f"Preview: {result.preview_text}")

            # Выводим информацию о сгенерированных шортах
            if result.generated_shorts:
                print(f"\nGenerated shorts:")
                for i, short_path in enumerate(result.generated_shorts, 1):
                    print(f"  {i}. {os.path.basename(short_path)}")

            # Сохраняем JSON результат
            import json
            import os
            from datetime import datetime

            output_dir = "film_analysis_results"
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"film_analysis_{result.video_id}_{timestamp}.json")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'video_id': result.video_id,
                    'duration': result.duration,
                    'keep_ranges': result.keep_ranges,
                    'scores': result.scores,
                    'preview_text': result.preview_text,
                    'risks': result.risks,
                    'metadata': result.metadata,
                    'generated_shorts': result.generated_shorts
                }, f, ensure_ascii=False, indent=2)

            print(f"Results saved to: {output_file}")

            # Устанавливаем output для совместимости с остальной логикой
            output = result.generated_shorts if result.generated_shorts else None
        else:
            print("\nFilm analysis failed!")
            output = None
    else:
        print("Invalid choice")
        output = None

    if output:
        # If output is a list (multiple shorts generated)
        if isinstance(output, list):
            print(f"\nSuccess! Output saved to:")
            for path in output:
                print(f"- {path}")
        else: # Should not happen with current logic, but handle just in case
            print(f"\nSuccess! Output saved to: {output}")
    else:
        print("\nProcessing failed or no shorts generated!")
