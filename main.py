from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extractAudio, crop_video, burn_captions, crop_bottom_video, animate_captions, get_video_dimensions
from Components.Transcription import transcribeAudio, transcribe_word_level_full
from Components.LanguageTasks import GetHighlights, build_transcription_prompt
from Components.FaceCrop import crop_to_vertical_average_face
from Components.Database import VideoDatabase
from dataclasses import dataclass, field
from typing import Optional, List
import os
import traceback
from Components.config import get_config, AppConfig

# Load config once
cfg = get_config()
print(f"Конфиг загружен: shorts_dir={cfg.processing.shorts_dir}, model={cfg.llm.model_name}")

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


def resolve_video_source(ctx: ProcessingContext) -> bool:
    """Определяет источник видео (URL/локальный), учитывает кэш БД, выставляет ctx.video_path и ctx.video_id."""
    if not ctx.url and not ctx.local_path:
        print("Error: Must provide either URL or local path")
        return False

    video_path = None
    video_id = None

    if ctx.url:
        print(f"Processing YouTube URL: {ctx.url}")
        cached_data = ctx.db.get_cached_processing(youtube_url=ctx.url)
        if cached_data:
            print("Found cached video from URL!")
            video_path = cached_data["video"][2]
            video_id = cached_data["video"][0]
            if not os.path.exists(video_path):
                print(f"Cached video path not found: {video_path}. Re-downloading.")
                video_path = None
                video_id = None
        if not video_path:
            video_path = download_youtube_video(ctx.url)
            if not video_path:
                print("Failed to download video")
                return False
            if not video_path.lower().endswith('.mp4'):
                base, _ = os.path.splitext(video_path)
                new_path = base + ".mp4"
                try:
                    os.rename(video_path, new_path)
                    video_path = new_path
                    print(f"Renamed downloaded file to: {video_path}")
                except OSError as e:
                    print(f"Error renaming file to mp4: {e}. Trying conversion.")
                    pass
    else:
        print(f"Processing local file: {ctx.local_path}")
        if not os.path.exists(ctx.local_path):
            print("Error: Local file does not exist")
            return False
        video_path = ctx.local_path
        cached_data = ctx.db.get_cached_processing(local_path=ctx.local_path)
        if cached_data:
            print("Found cached local video!")
            video_id = cached_data["video"][0]

    if not video_path or not os.path.exists(video_path):
        print("No valid video path obtained or file does not exist.")
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


def transcribe_or_load(ctx: ProcessingContext) -> bool:
    """Грузит транскрипцию из БД или вызывает Whisper; сохраняет полную JSON и сегменты в ctx."""
    transcriptions = None
    if ctx.video_id:
        transcriptions = ctx.db.get_transcription(ctx.video_id)
        if transcriptions:
            print("Using cached segment transcription")

    if not transcriptions:
        print("Generating new segment transcription")
        transcriptions = transcribeAudio(ctx.audio_path)
        if transcriptions:
            ctx.db.add_transcription(ctx.video_id, transcriptions)
            try:
                base = sanitize_base_name(os.path.splitext(os.path.basename(ctx.video_path))[0])
                payload = to_full_segments_payload(transcriptions)
                target = build_transcriptions_dir() / f"{base}_full_segments.json"
                save_json_safely(payload, target)
            except Exception:
                pass
        else:
            print("Segment-level transcription failed. Cannot proceed.")
            return False

    segments: List[dict] = []
    for item in (transcriptions or []):
        try:
            text, start, end = item
        except Exception:
            text = str(item.get("text", ""))
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
        segments.append({"text": str(text), "start": float(start), "end": float(end), "speaker": None})
    ctx.transcription_segments = segments
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
                print(f"[WordLevel] Adjusted stop from {prev_stop:.2f}s to {adjusted_stop:.2f}s based on last word end")
        # Гарантия: stop > start
        if adjusted_stop <= start:
            adjusted_stop = start + 0.1

        print(f"\n--- Processing Highlight {seq}/{total}: Start={start:.2f}s, End={stop:.2f}s (effective end {adjusted_stop:.2f}s) ---")
        if isinstance(item, dict) and "caption_with_hashtags" in item:
            print(f"Caption: {item['caption_with_hashtags']}")

        # --- Define File Paths ---
        base_name = os.path.splitext(os.path.basename(ctx.video_path))[0]
        output_base = f"{base_name}_highlight_{seq}"
        temp_segment = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_temp_segment.mp4")
        cropped_vertical_temp = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_vertical_temp.mp4")
        cropped_vertical_final = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_vertical_final.mp4")
        final_output_with_captions = os.path.join(SHORTS_DIR, f"{output_base}_final.mp4")
        if USE_ANIMATED_CAPTIONS:
            segment_audio_path = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_temp_audio.wav")

        # 1. Extract Segment (Video + Audio, Original Aspect Ratio)
        print("1. Extracting segment...")
        extract_success = crop_video(ctx.video_path, temp_segment, start, adjusted_stop, ctx.initial_width, ctx.initial_height)
        if not extract_success:
            print(f"Failed step 1 for highlight {seq}. Skipping.")
            if os.path.exists(temp_segment):
                try:
                    os.remove(temp_segment)
                except Exception as clean_e:
                    print(f"Warning: Could not remove temp segment file: {clean_e}")
            return None

        # --- CHECK DIMENSIONS: Segment ---
        print("\n--- Checking Segment Video Dimensions ---")
        segment_width, segment_height = get_video_dimensions(temp_segment)
        if segment_width is None or segment_height is None:
            print("Error: Could not determine segment video dimensions. Skipping highlight.")
            if os.path.exists(temp_segment):
                try:
                    os.remove(temp_segment)
                except Exception as clean_e:
                    print(f"Warning: Could not remove temp segment file: {clean_e}")
            return None
        if segment_width != ctx.initial_width or segment_height != ctx.initial_height:
            print(f"Warning: Segment dimensions ({segment_width}x{segment_height}) differ from initial ({ctx.initial_width}x{ctx.initial_height}).")
        print("--- Segment Check Done ---")

        # 2. Create Vertical Crop (Based on Average Face Position)
        print("2. Creating average face centered vertical crop...")
        vert_crop_path = crop_to_vertical_average_face(temp_segment, cropped_vertical_temp)
        if not vert_crop_path:
            print(f"Failed step 2 (average face crop) for highlight {seq}. Skipping.")
            if os.path.exists(temp_segment):
                try:
                    os.remove(temp_segment)
                except Exception as clean_e:
                    print(f"Warning: Could not remove temp segment file: {clean_e}")
            if os.path.exists(cropped_vertical_temp):
                try:
                    os.remove(cropped_vertical_temp)
                except Exception as clean_e:
                    print(f"Warning: Could not remove temp vertical crop file: {clean_e}")
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
                captioning_success = animate_captions(cropped_vertical_final, temp_segment, transcription_result, final_output_with_captions)
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
            captioning_success = burn_captions(cropped_vertical_final, temp_segment, transcriptions_legacy, start, adjusted_stop, final_output_with_captions)

        # 5. Handle Captioning Result
        if not captioning_success:
            print(f"Animated caption generation failed for highlight {seq}. Attempting ASS burn fallback...")
            transcriptions_legacy = [[
                str(seg.get("text", "")),
                float(seg.get("start", 0.0)),
                float(seg.get("end", 0.0)),
            ] for seg in (ctx.transcription_segments or [])]
            fallback_success = burn_captions(cropped_vertical_final, temp_segment, transcriptions_legacy, start, adjusted_stop, final_output_with_captions)
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

        print(f"Successfully processed highlight {seq}.")
        ctx.outputs.append(final_output_with_captions)
        print(f"Saving highlight {seq} info to database: {final_output_with_captions}")

        segment_text = item.get('segment_text', '') if isinstance(item, dict) else ''
        caption = item.get('caption_with_hashtags', '') if isinstance(item, dict) else ''

        ctx.db.add_highlight(
            ctx.video_id,
            start,
            adjusted_stop,
            final_output_with_captions,
            segment_text=segment_text,
            caption_with_hashtags=caption
        )

        # --- Cleanup Intermediate Files ---
        print("Cleaning up intermediate files for this highlight...")
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


def process_video(url: str = None, local_path: str = None):
    """
    Координатор пайплайна. Публичная сигнатура сохранена.
    Возвращает список путей к сгенерированным клипам или None при ошибке/пустом результате.
    """
    ctx = init_context(url, local_path)
    if not resolve_video_source(ctx):
        return None
    if not validate_dimensions(ctx):
        return None
    if not ensure_audio(ctx):
        return None
    if not transcribe_or_load(ctx):
        return None

    # Единая словная транскрипция на весь ролик (один раз)
    if ctx.word_level_transcription is None:
        print("\n[WordLevel] Starting global word-level transcription for the full audio...")
        wl = transcribe_word_level_full(ctx.audio_path)
        if wl:
            ctx.word_level_transcription = wl
            print("[WordLevel] Global word-level transcription computed and cached in context.")
        else:
            print("[WordLevel][WARN] Failed to compute global word-level transcription; proceeding without it.")

    prepare_transcript_text(ctx)

    try:
        highlights = fetch_highlights(ctx)
        if not highlights or len(highlights) == 0:
            print("No valid highlights found")
            return None

        outputs = process_all_highlights(ctx, highlights)
        if not outputs:
            return None
        return outputs
    except Exception as e:
        print(f"Error in overall highlight processing: {str(e)}")
        traceback.print_exc()
        return None


def prepare_words_for_segment(full_word_level_transcription: dict, start: float, stop: float) -> dict:
    """
    Подготавливает подмножество слов для одного сегмента [start, stop] из глобальной
    словной транскрипции и нормализует их таймкоды к координатам сегмента.

    Вход:
    - full_word_level_transcription: dict в формате
      {
        "segments": [
          {
            "words": [
              {"start": float, "end": float, "text"/"word": str, ...},
              ...
            ],
            ...
          },
          ...
        ]
      }
    - start: абсолютное время начала сегмента, секунды (float)
    - stop:  абсолютное время конца сегмента, секунды (float)

    Логика отбора и нормализации:
    - Выбираются только те слова, которые пересекаются с интервалом [start, stop]:
      word.start < stop и word.end > start.
    - Нормализация в координаты сегмента:
      start_rel = max(0.0, word.start - start)
      end_rel   = max(start_rel, word.end - start)
    - end_rel дополнительно ограничивается длительностью сегмента (stop - start), чтобы
      «хвост» слова за границей сегмента был обрезан.
    - Слова без числовых start/end пропускаются.
    - Результат сортируется по (start_rel, end_rel).

    Возврат:
    Структура совместимая с animate_captions:
    {
      "segments": [{
        "start": 0.0,
        "end": stop - start,
        "text": "segment",
        "words": [
          {"start": start_rel, "end": end_rel, "text": word_text}, ...
        ]
      }]}
    
    Предположения:
    - Функция чистая: не изменяет входные данные, не имеет побочных эффектов.
    - При некорректном входе возвращается сегмент с пустым списком слов и корректной длительностью.
    """
    try:
        seg_duration = max(0.0, float(stop) - float(start))
    except Exception:
        # На всякий случай, если приведение типов не удалось
        seg_duration = max(0.0, (stop or 0.0) - (start or 0.0))

    words_out = []
    try:
        segments_wl = []
        if isinstance(full_word_level_transcription, dict):
            segments_wl = full_word_level_transcription.get("segments", []) or []
        for seg_wl in segments_wl:
            # Достаём список слов из dict или объекта с атрибутом .words
            seg_words = seg_wl.get("words", []) if isinstance(seg_wl, dict) else getattr(seg_wl, "words", []) or []
            if not seg_words:
                continue
            for w in seg_words:
                if isinstance(w, dict):
                    s = _to_float(w.get("start", None), None)
                    e = _to_float(w.get("end", None), None)
                    txt = w.get("text", w.get("word", ""))
                else:
                    s = _to_float(getattr(w, "start", None), None)
                    e = _to_float(getattr(w, "end", None), None)
                    txt = getattr(w, "text", getattr(w, "word", ""))
                # Пропускаем некорректные таймкоды
                if s is None or e is None:
                    continue
                # Фильтр пересечения [start, stop]
                if e > start and s < stop:
                    start_rel = max(0.0, s - start)
                    end_rel = max(start_rel, e - start)
                    # Обрезка по длительности сегмента
                    if end_rel > seg_duration:
                        end_rel = seg_duration
                        if end_rel < start_rel:
                            start_rel = end_rel
                    words_out.append({"start": start_rel, "end": end_rel, "text": str(txt)})
        # Сортировка стабильна для предсказуемости
        words_out.sort(key=lambda x: (x["start"], x["end"]))
    except Exception:
        # В случае неожиданных проблем вернём пустой список слов, сохранив длительность
        words_out = []

    return {
        "segments": [{
            "start": 0.0,
            "end": seg_duration,
            "text": "segment",
            "words": words_out
        }]}
    

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
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        url = input("Enter YouTube URL: ")
        output = process_video(url=url)
    elif choice == "2":
        local_file = input("Enter path to local video file: ")
        output = process_video(local_path=local_file)
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
