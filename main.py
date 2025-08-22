from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extractAudio, get_video_dimensions, process_frame_for_vertical_short
from Components.Transcription import transcribe_unified
from faster_whisper import WhisperModel
import torch
import json
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
    """
    Sanitize filename to create safe base name for file operations.

    Args:
        name: Original filename or path

    Returns:
        Sanitized base name safe for file operations
    """
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
    """Convert value to float with error handling."""
    try:
        return float(val)
    except (ValueError, TypeError):
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


def process_highlight(ctx: ProcessingContext, item) -> Optional[str]:
    """
    Обрабатывает один хайлайт, используя оптимизированный однопроходный конвейер.
    Возвращает путь к финальному файлу или None в случае ошибки.
    """
    seq = int(item.get("_seq", 1)) if isinstance(item, dict) else 1
    total = int(item.get("_total", 1)) if isinstance(item, dict) else 1

    try:
        start = float(item["start"])
        stop = float(item["end"])

        # Корректировка stop по фактическому окончанию последнего полного слова
        adjusted_stop = stop
        if getattr(ctx, "word_level_transcription", None):
            last_word_end = find_last_word_end_time(ctx.word_level_transcription, stop)
            if last_word_end is not None and last_word_end > stop:
                adjusted_stop = last_word_end
                print(f"[WordLevel] Adjusted stop from {stop:.2f}s to {adjusted_stop:.2f}s")
        
        if adjusted_stop <= start:
            adjusted_stop = start + 0.1

        print(f"\n--- Processing Highlight {seq}/{total}: Start={start:.2f}s, End={adjusted_stop:.2f}s ---")
        if isinstance(item, dict) and "caption_with_hashtags" in item:
            print(f"Caption: {item['caption_with_hashtags']}")

        # --- Подготовка путей и параметров ---
        base_name = os.path.splitext(os.path.basename(ctx.video_path))[0]
        output_base = f"{base_name}_highlight_{seq}"
        final_output_path = os.path.join(SHORTS_DIR, f"{output_base}_final.mp4")

        # --- Подготовка подмножества слов для субтитров ---
        # Эта логика теперь инкапсулирована в create_ass_file, но нам нужно передать
        # полную транскрипцию.
        word_transcription_for_segment = ctx.word_level_transcription
        if not word_transcription_for_segment:
             print("[WARN] No word-level transcription available. Captions may not be generated.")
             word_transcription_for_segment = {"segments": []} # Fallback to empty

        # --- Вызов единой функции обработки ---
        success = process_frame_for_vertical_short(
            source_video_path=ctx.video_path,
            output_path=final_output_path,
            start_time=start,
            end_time=adjusted_stop,
            word_level_transcription=word_transcription_for_segment,
            crop_bottom_percent=CROP_PERCENTAGE_BOTTOM,
            face_cascade_path='haarcascade_frontalface_default.xml'
        )

        if not success:
            print(f"Failed to process highlight {seq} with the one-pass method. Skipping.")
            return None

        # --- Сохранение результатов в БД ---
        print(f"Successfully processed highlight {seq}. Output: {final_output_path}")
        ctx.outputs.append(final_output_path)
        
        segment_text = item.get('segment_text', '')
        caption = item.get('caption_with_hashtags', '')

        ctx.db.add_highlight(
            ctx.video_id,
            start,
            adjusted_stop,
            final_output_path,
            segment_text=segment_text,
            caption_with_hashtags=caption
        )
        
        return final_output_path

    except Exception:
        print(f"\n--- Error processing highlight {seq} --- ")
        traceback.print_exc()
        print("Continuing to next highlight if available.")
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
    """Выбирает оптимальные параметры для модели Whisper."""
    try:
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    device = "cuda" if has_cuda else "cpu"
    model_size = "large-v3" if has_cuda else "small"
    compute_type = "float16" if has_cuda else "int8"
    cpu_threads = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    return model_size, device, compute_type, cpu_threads


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

    # --- Загрузка модели Whisper ---
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
    if not run_unified_transcription(ctx, model):
        return None

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
