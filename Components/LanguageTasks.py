from google import genai
from typing import TypedDict, List, Optional, Any, Tuple, Dict
import json
import os
import re # Import regex for parsing transcription
import time
from dotenv import load_dotenv
from google.genai import types
from Components.config import get_config

# Optional imports for Google API exceptions (rate limit handling)
try:
    from google.api_core.exceptions import ResourceExhausted as GoogleResourceExhausted  # type: ignore
except Exception:
    GoogleResourceExhausted = None  # type: ignore

try:
    from google.api_core import exceptions as google_exceptions  # type: ignore
except Exception:
    google_exceptions = None  # type: ignore

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "Google API key not found. Make sure it is defined in the .env file."
    )

client = genai.Client(
        api_key=GOOGLE_API_KEY,
    )
# Load configuration (cached)
cfg = get_config()
# Consider using a more capable model if generating descriptions needs more nuance
# model = genai.GenerativeModel("gemini-1.5-flash") # Example alternative
model = cfg.llm.model_name

def build_transcription_prompt(segments: list[dict]) -> str:
    """
    Собирает строку транскрипции для LLM из списка сегментов.

    Вход:
    - segments: список словарей со следующими ключами (минимально необходимые):
        - "start": float
        - "end": float
        - "text": str
        - опционально: "speaker" | "name" | "id" — будет использовано вместо "Speaker", если непусто.

    Формат каждой строки:
    "[{start:.2f}] SpeakerName: {text} [{end:.2f}]"

    Возврат:
    - Одна большая строка с символом новой строки после каждой записи. Побочных эффектов нет.
    """
    lines: list[str] = []
    for seg in (segments or []):
        try:
            if isinstance(seg, dict):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                text = str(seg.get("text", "") or "")
                speaker_val = None
                for key in ("speaker", "name", "id"):
                    v = seg.get(key, None)
                    if v is not None and str(v).strip():
                        speaker_val = str(v).strip()
                        break
            else:
                start = float(getattr(seg, "start", 0.0))
                end = float(getattr(seg, "end", 0.0))
                text = str(getattr(seg, "text", "") or "")
                speaker_val = None
                for key in ("speaker", "name", "id"):
                    if hasattr(seg, key):
                        v = getattr(seg, key)
                        if v is not None and str(v).strip():
                            speaker_val = str(v).strip()
                            break

            speaker_label = speaker_val if speaker_val else "Speaker"
            line = f"[{start:.2f}] {speaker_label}: {text.strip()} [{end:.2f}]"
            lines.append(line)
        except Exception:
            # Любые странности сегмента — пропускаем строку, не прерывая пайплайн
            continue
    return "\n".join(lines) + ("\n" if lines else "")
# --- Rate limit handling utilities and wrapper ---

def parse_retry_delay_seconds(error: Exception | str) -> Optional[int]:
    """
    Пытается извлечь задержку повторной попытки (в секундах) из текста ошибки.
    Поддерживаемые форматы:
    - Retry-After: 28
    - retry-after: 28
    - "retryDelay": "28s"
    - retryDelay: 28s
    Возвращает целое количество секунд или None, если не удалось распарсить.
    """
    text = ""
    try:
        if isinstance(error, Exception):
            parts = [str(error), repr(error)]
            for attr in ("message", "details", "args"):
                val = getattr(error, attr, None)
                if val:
                    parts.append(str(val))
            text = " | ".join(parts)
        else:
            text = str(error)
    except Exception:
        text = str(error)

    patterns = [
        r'(?i)(?:retry[- ]?after|retryDelay)"?:?\s*"?(\d+)\s*s?',  # Retry-After: 28  or retryDelay: "28s"
        r'(?i)"retryDelay"\s*:\s*"?(\d+)\s*s"?'                    # "retryDelay": "28s"
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None


def _is_resource_exhausted_error(err: Exception) -> bool:
    """Возвращает True, если ошибка соответствует лимиту API (ResourceExhausted)."""
    try:
        if 'GoogleResourceExhausted' in globals() and GoogleResourceExhausted is not None and isinstance(err, GoogleResourceExhausted):  # type: ignore
            return True
    except Exception:
        pass
    try:
        if 'google_exceptions' in globals() and google_exceptions is not None:
            ResExh = getattr(google_exceptions, "ResourceExhausted", None)
            if ResExh is not None and isinstance(err, ResExh):
                return True
    except Exception:
        pass
    text = f"{type(err).__name__}: {err}"
    return ("ResourceExhausted" in text) or ("RESOURCE_EXHAUSTED" in text) or ("rate limit" in text.lower())


def call_llm_with_retry(
    system_instruction: Optional[str],
    content: List | str,
    generation_config,
    model: Optional[str] = None,
    max_api_attempts: int = 3,
):
    """
    Выполняет вызов client.models.generate_content с централизованной обработкой лимитов API.

    Логирование:
    - При перехвате лимита и наличии retryDelay:
      "Лимит API обработан. Выполняю паузу на X секунд перед попыткой #Y."
    - Если retryDelay извлечь не удалось:
      "Не удалось извлечь retryDelay. Попытки прекращены."

    Стратегия:
    - Повторяет запрос не более max_api_attempts раз, делая паузу X секунд, если retryDelay присутствует.
    - При отсутствии retryDelay немедленно прекращает дальнейшие попытки и пробрасывает исключение.
    - Другие исключения пробрасываются без изменений.
    """
    model_to_use = model or globals().get("model")
    # Нормализуем contents
    if isinstance(content, list):
        contents = content
    else:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=str(content))])]

    last_err: Optional[Exception] = None

    for api_try in range(1, max_api_attempts + 1):
        try:
            # system_instruction ожидается внутри generation_config; параметр system_instruction оставлен для совместимости.
            return client.models.generate_content(
                model=model_to_use,
                contents=contents,
                config=generation_config,
            )
        except Exception as e:
            last_err = e
            if _is_resource_exhausted_error(e):
                delay = parse_retry_delay_seconds(e)
                if delay is None:
                    print("Не удалось извлечь retryDelay. Попытки прекращены.")
                    raise
                if api_try < max_api_attempts:
                    print(f"Лимит API обработан. Выполняю паузу на {delay} секунд перед попыткой #{api_try+1}.")
                    time.sleep(delay)
                    continue
                # Достигнут лимит попыток API — пробрасываем исключение без дополнительного лога.
                raise
            else:
                raise

    if last_err is not None:
        raise last_err

# Вспомогательная функция: безопасная сборка конфигурации генерации с поддержкой Thinking (если доступно в SDK)
def make_generation_config(system_instruction_text: str, temperature: float = 0.2) -> types.GenerateContentConfig:
    """
    Собирает GenerateContentConfig согласно документации google-genai:
    - system_instruction: строка (может быть Content, но здесь — строка).
    - response_mime_type='application/json' для строгого JSON.
    - thinking_config: types.ThinkingConfig(thinking_budget=-1, include_thoughts=False) для Gemini 2.5.
      Если класс отсутствует или не поддерживается текущей версией SDK — используем конфигурацию без thinking_config.
    """
    base_kwargs = dict(
        temperature=temperature,
        response_mime_type="application/json",
        system_instruction=system_instruction_text,
    )

    # Предпочтительный путь (Python): вложенный thinking_config с корректными snake_case полями.
    ThinkingConfig = getattr(types, "ThinkingConfig", None)
    if ThinkingConfig is not None:
        try:
            return types.GenerateContentConfig(
                **base_kwargs,
                thinking_config=ThinkingConfig(
                    thinking_budget=-1,      # динамическое мышление
                    include_thoughts=False,  # не включать конспекты мыслей в ответ
                ),
            )
        except Exception:
            # Если валидация SDK не пропускает thinking_config, откатываемся на конфиг без мышления.
            print("Предупреждение: ThinkingConfig недоступен/не совместим; продолжаю без thinking.")
            return types.GenerateContentConfig(**base_kwargs)

    # Фолбэк: нет ThinkingConfig в текущем SDK — работаем без мышления.
    return types.GenerateContentConfig(**base_kwargs)


class Message(TypedDict):
    role: str
    content: str


class HighlightSegment(TypedDict):
    start: float
    end: float

# New type for the enriched highlight data
class EnrichedHighlightData(TypedDict):
    start: float
    end: float
    caption_with_hashtags: str
    segment_text: str # Store the text used for generation
    title: Optional[str]
    description: Optional[str]
    hashtags: Optional[List[str]]


def validate_highlight(highlight: HighlightSegment) -> bool:
    """Validate a single highlight segment's time duration and format."""
    try:
        if not all(key in highlight for key in ["start", "end"]):
            print(f"Validation Fail: Missing 'start' or 'end' key in {highlight}")
            return False

        start = float(highlight["start"])
        end = float(highlight["end"])
        duration = end - start

        # Check for valid duration (configured range)
        min_duration = float(cfg.llm.highlight_min_sec)
        max_duration = float(cfg.llm.highlight_max_sec)

        if not (min_duration <= duration <= max_duration):
            print(f"Validation Fail: Duration {duration:.2f}s out of range [~{min_duration:.0f}s, ~{max_duration:.0f}s] for {highlight}")
            return False

        # Check for valid ordering
        if start >= end:
            print(f"Validation Fail: Start time {start} >= end time {end} for {highlight}")
            return False

        return True
    except (ValueError, TypeError) as e:
        print(f"Validation Fail: Invalid type or value in {highlight} - {e}")
        return False


def validate_highlights(highlights: List[HighlightSegment]) -> bool:
    """Validate all highlights and check for overlaps."""
    if not highlights:
        print("Validation: No highlights provided.")
        return False

    # Validate each individual highlight (already checks duration)
    if not all(validate_highlight(h) for h in highlights):
        # Specific errors printed within validate_highlight
        print("Validation: One or more highlights failed individual checks.")
        return False

    # Check for overlapping segments
    sorted_highlights = sorted(highlights, key=lambda x: float(x["start"]))
    for i in range(len(sorted_highlights) - 1):
        if float(sorted_highlights[i]["end"]) > float(
            sorted_highlights[i + 1]["start"]
        ):
            print(f"Validation Fail: Overlap detected between {sorted_highlights[i]} and {sorted_highlights[i+1]}")
            return False

    return True


def extract_highlights(
    transcription: str, max_attempts: int = 3
) -> List[HighlightSegment]:
    """Extracts highlight time segments from transcription, validates, checks overlaps, with retry logic."""
    # System instruction based on Google AI Studio code
    system_instruction_text = f"""
Ты — креативный ИИ-ассистент, который помогает находить лучшие моменты в видео для создания коротких роликов (shorts). Проанализируй транскрипцию и выдели несколько наиболее интересных и содержательных фрагментов.

Твоя задача — вернуть JSON-массив объектов. Каждый объект должен представлять один хайлайт и содержать:
- "start": время начала (float)
- "end": время окончания (float)

Пожалуйста, верни только JSON, без каких-либо дополнительных пояснений или текста.

Рекомендации по выбору хайлайта:
- Содержит ключевую мысль, яркий момент, вопрос или вывод.
- Является относительно законченным по смыслу.
- Длительность (end - start) в идеале составляет от {cfg.llm.highlight_min_sec} до {cfg.llm.highlight_max_sec} секунд. Смысл важнее точной длительности.
- Сегменты не должны пересекаться. Постарайся найти до {cfg.llm.max_highlights} наиболее подходящих сегментов.

Точность таймкодов:
- Используй временные метки, которые присутствуют в транскрипте. Не придумывай свои.

Пример твоего ответа:
[
  {{"start": 8.96, "end": 42.20}},
  {{"start": 115.08, "end": 156.12}}
]
"""

    # Define generation config based on AI Studio code
    generation_config = make_generation_config(system_instruction_text, temperature=cfg.llm.temperature_highlights)

    effective_attempts = cfg.llm.max_attempts_highlights if max_attempts == 3 else max_attempts
    for attempt in range(effective_attempts):
        print(f"\nПопытка {attempt + 1}: генерация и валидация тайм-сегментов для хайлайтов...")
        try:
            # Structure the prompt and system instruction for generate_content
            user_prompt_text = f"Transcription:\n{transcription}"
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt_text)])]
            # Use the global model but with the new config
            response = call_llm_with_retry(
                system_instruction=None,
                content=contents,
                generation_config=generation_config,
                model=model,
            )

            # Basic safety check for response content
            if not response or not response.text:
                 print(f"Неудача на попытке {attempt + 1}: пустой ответ от LLM.")
                 continue
 
            print(f"Сырой ответ LLM на попытке {attempt + 1}:\n---\n{response.text}\n---")
             # Extract JSON from response
            response_text = response.text
            # Handle potential markdown code blocks
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
            if match:
                json_string = match.group(1).strip()
            else:
                 # Assume the whole text is JSON if no markdown block found
                 json_string = response_text.strip()

            raw_highlights = json.loads(json_string)

            if not isinstance(raw_highlights, list):
                 print(f"Неудача на попытке {attempt + 1}: ответ LLM не является JSON‑массивом.")
                 print(f"Сырой ответ LLM: {response.text}")
                 continue

            # --- Улучшенная валидация и обработка пересечений ---
            print(f"LLM вернула {len(raw_highlights)} потенциальных сегментов. Начинаю валидацию.")
            
            # 1. Индивидуальная валидация (формат, длительность)
            individually_valid_highlights = []
            for h in raw_highlights:
                if validate_highlight(h):
                    individually_valid_highlights.append(h)
                else:
                    print(f"Отбракован сегмент из-за индивидуальной валидации: {h}")

            print(f"После проверки формата и длительности осталось {len(individually_valid_highlights)} валидных сегментов.")

            if not individually_valid_highlights:
                print("Ни один из сегментов не прошел индивидуальную валидацию. Следующая попытка.")
                continue

            # 2. Обработка пересечений: просто отбрасываем пересекающиеся
            sorted_highlights = sorted(individually_valid_highlights, key=lambda x: float(x.get("start", 0.0)))
            
            non_overlapping_highlights = []
            if sorted_highlights:
                last_valid_segment = sorted_highlights[0]
                non_overlapping_highlights.append(last_valid_segment)
                
                for i in range(1, len(sorted_highlights)):
                    current_segment = sorted_highlights[i]
                    last_end = float(last_valid_segment.get("end", 0.0))
                    current_start = float(current_segment.get("start", 0.0))

                    if current_start < last_end:
                        print(f"Обнаружено и отброшено пересечение: {current_segment} (start: {current_start}) начинается раньше, чем заканчивается предыдущий (end: {last_end}).")
                        continue
                    else:
                        non_overlapping_highlights.append(current_segment)
                        last_valid_segment = current_segment

            if not non_overlapping_highlights:
                print("После обработки пересечений не осталось валидных сегментов.")
                continue

            # If we reach here, we have a non-empty list of valid, non-overlapping highlights
            print(f"Успех на попытке {attempt + 1}. Найдено валидных сегментов: {len(non_overlapping_highlights)}.")
            # Apply max_highlights cap from config
            try:
                max_h = int(cfg.llm.max_highlights)
                if max_h > 0:
                    return non_overlapping_highlights[:max_h]
            except Exception:
                pass
            return non_overlapping_highlights # Return the validated and sorted list

            # If we reach here, we have a non-empty list of valid, non-overlapping highlights

        except json.JSONDecodeError:
             print(f"Неудача на попытке {attempt + 1}: некорректный JSON от LLM.")
             if 'response_text' in locals(): print(f"Сырой ответ LLM: {response_text}")
             continue
        except Exception as e:
            if _is_resource_exhausted_error(e):
                # Обертка уже залогировала причину; прекращаем дальнейшие попытки этой функции
                break
            print(f"Неудача на попытке {attempt + 1}: непредвиденная ошибка: {str(e)}")
            if 'response_text' in locals():
                print(f"Сырой ответ LLM при ошибке: {response_text}")
            continue

    print("Достигнуто максимальное число попыток извлечения сегментов. Возвращаю пустой список.")
    return []


# --- New Functions ---

def extract_text_for_segment(transcription: str, start_time: float, end_time: float) -> str:
    """Extracts speaker text from transcription within a given time range."""
    segment_text = []
    # Regex to capture timestamp and text, robust to formats like:
    # [0.00] Speaker: Text [8.96]
    # [8.96] Text
    # [12.32] Speaker: Text
    # It captures the start time and the main text content.
    line_pattern = re.compile(r"^\s*\[\s*(\d+\.\d+)\s*\]\s*(.*?)(?:\s*\[\d+\.\d+\s*\])?$")

    lines = transcription.strip().splitlines() # Use splitlines for robustness
    for i, line in enumerate(lines):
        match = line_pattern.match(line)
        if match:
            try:
                timestamp = float(match.group(1))
                text_content = match.group(2).strip()

                # Remove speaker prefix like "Speaker X:" if present
                text_content = re.sub(r"^[Ss]peaker\s*\d*:\s*", "", text_content).strip()

                # Include lines starting within the time range
                if timestamp < end_time and timestamp >= start_time:
                    if text_content: # Avoid adding empty lines
                         segment_text.append(text_content)
            except (ValueError, IndexError):
                # Ignore lines that don't match the expected format
                continue
        # else: Line doesn't match pattern, ignore

    return "\n".join(segment_text)


def generate_description_and_hashtags(segment_text: str, max_attempts: int = 3) -> Optional[str]:
    """Generates a description with appended hashtags for a text segment using LLM."""
    if not segment_text or not segment_text.strip():
        print("Skipping description generation: Empty segment text provided.")
        return None

    system_prompt = """
    Тебе дан текстовый фрагмент короткого видеоролика (обычно 30–60 секунд).
    Твоя задача: вернуть ОДИН JSON-объект со строкой:
    1) Короткое, ёмкое и вовлекающее описание (1–2 предложения) содержимого клипа.
    2) Затем через пробел — 3–5 релевантных хэштегов, слитно, в нижнем регистре, начинаются с #, без пробелов.
    
    Формат ответа (строго, на английском ключе):
    {
        "caption_with_hashtags": "Твоё описание. #пример1 #пример2 #пример3"
    }
    
    Правила:
    - В ответе не должно быть ничего, кроме указанного JSON-объекта.
    - Хэштеги отражают тему клипа, без пробелов, латиницей/кириллицей допустимо.
    
    Пример ввода:
    "Одна из самых интересных областей — обработка видео. Посмотрим, как ИИ может автоматически находить хайлайты."
    
    Пример вывода:
    {
        "caption_with_hashtags": "Как ИИ находит лучшие моменты в видео — кратко и по делу! #ии #видео #хайлайты #машинноевобучение"
    }
    
    Верни ТОЛЬКО JSON-объект.
    """

    # Define generation config based on AI Studio code
    generation_config = make_generation_config(system_prompt, temperature=cfg.llm.temperature_metadata)

    effective_attempts = cfg.llm.max_attempts_metadata if max_attempts == 3 else max_attempts
    for attempt in range(effective_attempts):
        try:
            # Structure the prompt and system instruction for generate_content
            user_prompt_text = f"Segment Text:\n{segment_text}"
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt_text)])]
            
            # Use the global model with the new config
            response = call_llm_with_retry(
                system_instruction=None,
                content=contents,
                generation_config=generation_config,
                model=model,
            )

            if not response or not response.text:
                print(f"Неудача на попытке {attempt + 1}: пустой ответ от LLM для описания.")
                continue

            # Extract JSON from response
            response_text = response.text
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
            if match:
                json_string = match.group(1).strip()
            else:
                json_string = response_text.strip()

            data = json.loads(json_string)

            # Validate the structure and types
            if not isinstance(data, dict) or \
               "caption_with_hashtags" not in data or \
               not isinstance(data["caption_with_hashtags"], str):
                print(f"Неудача на попытке {attempt + 1}: некорректная структура или типы в JSON-ответе.")
                print(f"Сырой ответ LLM: {response_text}")
                continue

            return data["caption_with_hashtags"].strip()

        except json.JSONDecodeError:
            print(f"Неудача на попытке {attempt + 1}: некорректный JSON от LLM для описания.")
            if 'response_text' in locals(): print(f"Сырой ответ LLM: {response_text}")
            continue
        except Exception as e:
            if _is_resource_exhausted_error(e):
                break
            print(f"Неудача на попытке {attempt + 1}: непредвиденная ошибка при генерации описания: {str(e)}")
            if 'response_text' in locals():
                print(f"Сырой ответ LLM при ошибке: {response_text}")
            continue

    print("Достигнуто максимальное число попыток генерации описания/хэштегов. Возвращаю None.")
    return None

# --- Batch metadata generation ---
BATCH_METADATA_SYSTEM_PROMPT = """Ты — эксперт по SMM и продвижению на YouTube, специализирующийся на вирусных Shorts. Тебе на вход подается JSON-массив текстовых фрагментов из видео, каждый с уникальным `id`. Твоя задача — для каждого фрагмента создать оптимальный набор метаданных для максимального вовлечения и охвата.

Правила:
1. Твой ответ должен быть ИСКЛЮЧИТЕЛЬНО одним валидным JSON-массивом. Никакого текста до или после.
2. Для каждого входного объекта с `id` ты должен сгенерировать объект в выходном массиве с тем же `id` и тремя полями: `title`, `description` и `hashtags`.
3. title (заголовок): 40–70 символов, интригующий, задает вопрос или создает предвкушение. Обязательно использовать ключевые слова из текста.
4. description (описание): до 150 символов, кратко раскрывает суть, допускается призыв к действию.
5. hashtags (хэштеги): массив из 3–5 строк; первым ВСЕГДА `#shorts`; остальные — максимально релевантны теме фрагмента.

Пример Входа:
[{"id":"seg_1","text":"Сегодня обсудим, как автоматически находить лучшие моменты в видео..."}]

Пример Выхода:
[{"id":"seg_1","title":"Нейросеть находит лучшие моменты в видео?","description":"Смотрите, как ИИ анализирует ролики для создания шортсов.","hashtags":["#shorts","#ИИ","#нейросети","#видеомонтаж"]}]"""

def _build_retry_prompt(
    validation_tracker: Dict[str, Dict[str, Any]],
    items_to_retry: Optional[List[dict]] = None,
    *,
    max_snippet_len: int = 500
) -> str:
    """
    Строит user prompt для повторной отправки проблемных элементов.

    Параметры:
    - validation_tracker: словарь статусов вида {id: {"status","data","reason","original_item"}}
    - items_to_retry: список исходных элементов {"id","text"} для повтора; если None — берутся со статусами pending/failed
    - max_snippet_len: ограничение длины включаемого текста

    Возвращает:
    - Строку, которую следует передать как пользовательский prompt.
    """
    try:
        if items_to_retry is None:
            items_to_retry = []
            for _id, st in validation_tracker.items():
                if st and st.get("status") in ("pending", "failed"):
                    orig = st.get("original_item") or {}
                    if "id" not in orig:
                        orig = {**orig, "id": _id}
                    items_to_retry.append(orig)
    except Exception:
        items_to_retry = items_to_retry or []

    lines: List[str] = []
    lines.append("Нужно исправить ошибки для указанных id. Верни строго JSON-массив объектов с корректированными данными для этих id.")
    lines.append("")
    lines.append("Требования к каждому объекту ответа:")
    lines.append("• Сохраняй поле id без изменений.")
    lines.append("• Поля: title, description, hashtags.")
    lines.append("• title: 40–70 символов (после тримминга).")
    lines.append("• description: максимум 150 символов.")
    lines.append("• hashtags: массив из 3–5 строк; первый элемент строго '#shorts'; все элементы начинаются с '#'.")
    lines.append("")
    lines.append("Проблемные элементы (id, причина и исходный текст):")
    lines.append("")

    for it in items_to_retry:
        _id = str(it.get("id", ""))
        st = validation_tracker.get(_id, {}) or {}
        reason = st.get("reason") or "Причина не указана — см. требования валидации."
        text = str(it.get("text", "") or "")
        snippet = text.strip()
        if len(snippet) > max_snippet_len:
            snippet = snippet[:max_snippet_len].rstrip() + "..."
        lines.append(f"- id: {_id}")
        lines.append(f"  Причина ошибки: {reason}")
        lines.append("  Исходный текст:")
        lines.append(f'  """{snippet}"""')
        lines.append("")

    lines.append("Верни ТОЛЬКО JSON-массив следующего вида без пояснений:")
    lines.append('[{"id":"<id>","title":"...","description":"...","hashtags":["#shorts","..."]}]')
    return "\n".join(lines)

def generate_metadata_batch(items: list[dict], max_attempts: int = 3) -> list[dict]:
    """
    Пакетная генерация метаданных (title, description, hashtags) для сегментов с таргетированными повторами.

    Параметры:
    - items: список словарей вида {"id": str, "text": str}
    - max_attempts: максимальное число итераций (первая — весь батч, далее — только проблемные)

    Возврат:
    - список объектов {"id": str, "title": str, "description": str, "hashtags": list[str]} в порядке исходных items.

    Валидация (правила НЕ изменены):
    - title: 40–70 символов (после trim)
    - description: длина ≤ 150 символов
    - hashtags: массив длиной 3–5; первый элемент — "#shorts"; все элементы — строки, начинающиеся с "#".
    """
    if not items:
        print("Пакетная генерация метаданных: входных сегментов = 0")
        return []

    print(f"Пакетная генерация метаданных: входных сегментов = {len(items)}")

    # Первая итерация — как и раньше: системный промпт из константы
    generation_config_first = make_generation_config(
        BATCH_METADATA_SYSTEM_PROMPT,
        temperature=cfg.llm.temperature_metadata,
    )

    def _clean_space(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    def _extract_json_array(text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = text.strip()
        m = re.search(r"```json\s*([\s\S]*?)\s*```", t)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s*([\s\S]*?)\s*```", t)
        if m:
            return m.group(1).strip()
        m = re.search(r'"""([\s\S]*?)"""', t)
        if m:
            return m.group(1).strip()
        start = t.find("[")
        end = t.rfind("]")
        if start != -1 and end != -1 and start < end:
            return t[start:end+1].strip()
        return t

    def _validate_item(obj: dict, expected_id: str) -> Tuple[bool, Optional[str], Optional[dict]]:
        """
        Проверяет объект метаданных согласно неизменённым правилам.

        Возвращает:
        - success: bool
        - reason: текст причины ошибки (рус.), если неуспех
        - valid_data: нормализованный объект при успехе
        """
        if not isinstance(obj, dict):
            return False, "Элемент ответа должен быть JSON-объектом (dict).", None
        obj_id = str(obj.get("id", ""))
        if obj_id != expected_id:
            return False, f"Неверный id: ожидалось '{expected_id}', получено '{obj_id}'.", None

        title = _clean_space(str(obj.get("title", "")))
        description = _clean_space(str(obj.get("description", "")))
        hashtags = obj.get("hashtags", [])

        if not (40 <= len(title) <= 70):
            return False, f"Некорректная длина title: {len(title)} символов; требуется 40–70.", None
        if len(description) > 150:
            return False, f"Слишком длинный description: {len(description)} символов; максимум 150.", None
        if not isinstance(hashtags, list):
            return False, "hashtags должен быть списком из 3–5 строк.", None
        if not (3 <= len(hashtags) <= 5):
            return False, f"Некорректное количество хэштегов: {len(hashtags)}; требуется 3–5.", None
        if any(not isinstance(h, str) or not h.startswith("#") for h in hashtags):
            return False, "Все хэштеги должны быть строками и начинаться с '#'.", None
        if len(hashtags) == 0 or hashtags[0] != "#shorts":
            return False, "Первый хэштег должен быть '#shorts'.", None

        return True, None, {
            "id": expected_id,
            "title": title,
            "description": description,
            "hashtags": hashtags,
        }

    # Подготовка входных элементов и трекера валидации
    items_prepared: List[dict] = []
    ordered_ids: List[str] = []
    validation_tracker: Dict[str, Dict[str, Any]] = {}

    for idx, it in enumerate(items):
        safe = dict(it or {})
        id_val = safe.get("id")
        if id_val is None or str(id_val).strip() == "":
            id_val = f"item_{idx+1}"
            safe["id"] = id_val
        id_str = str(id_val)
        if "text" in safe:
            safe["text"] = str(safe["text"])
        items_prepared.append(safe)
        ordered_ids.append(id_str)
        validation_tracker[id_str] = {
            "status": "pending",       # "pending" | "success" | "failed"
            "data": None,              # валидные данные при успехе
            "reason": None,            # последняя причина отказа
            "original_item": safe,     # исходный элемент {"id","text",...}
        }

    N = len(items_prepared)
    effective_attempts = cfg.llm.max_attempts_metadata if max_attempts == 3 else max_attempts

    attempt = 0
    while attempt < effective_attempts:
        attempt += 1

        # Определяем поднабор для текущей попытки
        if attempt == 1:
            to_send = items_prepared
        else:
            to_send = [
                validation_tracker[_id]["original_item"]
                for _id in ordered_ids
                if validation_tracker[_id]["status"] in ("pending", "failed")
            ]

        if not to_send:
            # Все уже успешно провалидированы
            break

        print(f"Попытка {attempt} из {effective_attempts} для batch-метаданных ({len(to_send)} элементов)")
        try:
            if attempt == 1:
                # Первая отправка — как раньше: весь вход и основной модель
                user_payload = json.dumps(to_send, ensure_ascii=False)
                contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_payload)])]
                gen_cfg = generation_config_first
                model_to_use = model
            else:
                # Повторы — только проблемные элементы, со вспомогательным prompt и лёгкой моделью
                retry_prompt = _build_retry_prompt(validation_tracker, to_send)
                contents = [types.Content(role="user", parts=[types.Part.from_text(text=retry_prompt)])]
                # Системный промпт остаётся тем же, чтобы зафиксировать схему ответа
                gen_cfg = make_generation_config(BATCH_METADATA_SYSTEM_PROMPT, temperature=cfg.llm.temperature_metadata)
                model_to_use = "gemini-2.5-flash-lite"

            response = call_llm_with_retry(
                system_instruction=None,  # system_instruction уже в gen_cfg
                content=contents,
                generation_config=gen_cfg,
                model=model_to_use,
            )

            if not response or not getattr(response, "text", None):
                print(f"Неудача на попытке {attempt}: пустой ответ от LLM.")
                continue

            raw_text = response.text
            json_str = _extract_json_array(raw_text)
            data = json.loads(json_str)

            if not isinstance(data, list):
                print(f"Неудача на попытке {attempt}: ответ LLM не является JSON‑массивом.")
                continue

            # Соберём кандидатов по id
            out_by_id: Dict[str, Any] = {}
            for obj in data:
                if isinstance(obj, dict) and "id" in obj:
                    out_by_id[str(obj["id"])] = obj

            validated_this_attempt = 0
            expected_ids = [str(it.get("id")) for it in to_send]

            for expected_id in expected_ids:
                candidate = out_by_id.get(expected_id)
                if candidate is None:
                    validation_tracker[expected_id]["status"] = "failed"
                    validation_tracker[expected_id]["data"] = None
                    validation_tracker[expected_id]["reason"] = f"Модель не вернула объект для id '{expected_id}'."
                    continue

                ok, reason, normalized = _validate_item(candidate, expected_id)
                if ok and normalized:
                    validation_tracker[expected_id]["status"] = "success"
                    validation_tracker[expected_id]["data"] = normalized
                    validation_tracker[expected_id]["reason"] = None
                    validated_this_attempt += 1
                else:
                    validation_tracker[expected_id]["status"] = "failed"
                    validation_tracker[expected_id]["data"] = None
                    validation_tracker[expected_id]["reason"] = reason or "Нарушение правил валидации."

            total_success = sum(1 for st in validation_tracker.values() if st["status"] == "success")
            print(f"Batch-метаданные: успешно сгенерировано за попытку {validated_this_attempt}, всего валидных {total_success} из {N}")

            if total_success == N:
                break

        except json.JSONDecodeError:
            print(f"Неудача на попытке {attempt}: некорректный JSON от LLM для batch-метаданных.")
            if 'raw_text' in locals():
                print(f"Сырой ответ LLM: {raw_text}")
            continue
        except Exception as e:
            if _is_resource_exhausted_error(e):
                # Внутренняя обёртка уже залогировала и управляла паузами; прекращаем дальнейшие итерации
                break
            print(f"Неудача на попытке {attempt}: непредвиденная ошибка при batch-метаданных: {e}")
            if 'raw_text' in locals():
                print(f"Сырой ответ LLM при ошибке: {raw_text}")
            continue

    # Итоговая сборка по исходному порядку; для неуспехов — плейсхолдеры
    if any(st["status"] != "success" for st in validation_tracker.values()):
        print("Batch-метаданные: не все элементы прошли валидацию, для оставшихся будут использованы плейсхолдеры")

    results: List[dict] = []
    for _id in ordered_ids:
        st = validation_tracker[_id]
        if st["status"] == "success" and st["data"]:
            results.append(st["data"])
        else:
            results.append({"id": _id, "title": "", "description": "", "hashtags": ["#shorts"]})

    return results

# --- Updated Main Function ---

def GetHighlights(transcription: str) -> List[EnrichedHighlightData]:
    """
    Main function to get multiple highlight segments from transcription,
    each enriched with LLM-generated metadata (title, description, hashtags).
    Backward-compatible: caption_with_hashtags is preserved.
    """
    enriched_highlights = []
    try:
        # Clean and validate the input transcription
        if not transcription or not transcription.strip():
            print("Ошибка: передана пустая транскрипция.")
            return []

        # 1. Extract highlight time segments
        highlight_segments = extract_highlights(transcription.strip())

        if not highlight_segments:
            print("Не удалось извлечь валидные тайм‑сегменты для хайлайтов.")
            return []

        # 2. Extract text and prepare batch items
        items = []
        mapping = []

        for idx, segment in enumerate(highlight_segments, start=1):
            # Convert string timestamps to floats first
            try:
                start_time = float(segment["start"])
                end_time = float(segment["end"])
            except ValueError:
                print(f"Предупреждение: не удалось преобразовать таймкоды в float для сегмента {segment}. Пропускаю.")
                continue

            # Extract text for this segment
            segment_text = extract_text_for_segment(transcription, start_time, end_time)

            if not segment_text.strip():
                print("Предупреждение: для этого сегмента не извлечён текст. Пропускаю генерацию метаданных.")
                continue

            seg_id = f"seg_{len(items)+1}"
            items.append({"id": seg_id, "text": segment_text})
            mapping.append((seg_id, start_time, end_time, segment_text))

        if not items:
            print("Не удалось подготовить ни одного текстового сегмента для пакетной генерации.")
            return []

        print(f"\nПерехожу к пакетной генерации метаданных для {len(items)} сегментов...")
        batch_meta = generate_metadata_batch(items)
        meta_by_id = {str(m.get("id")): m for m in (batch_meta or []) if isinstance(m, dict)}

        # 3. Assemble enriched highlights
        for seg_id, start_time, end_time, segment_text in mapping:
            meta = meta_by_id.get(seg_id, {})
            title = str(meta.get("title", "") or "").strip()
            description = str(meta.get("description", "") or "").strip()
            hashtags = meta.get("hashtags", None)
            if not isinstance(hashtags, list) or not all(isinstance(h, str) for h in hashtags):
                hashtags = ["#shorts"]

            # Build backward-compatible caption_with_hashtags
            base_caption = ""
            if title and description:
                base_caption = f"{title} — {description}"
            elif title:
                base_caption = title
            elif description:
                base_caption = description
            caption_with_hashtags = base_caption
            if hashtags:
                caption_with_hashtags = f"{base_caption}\n{' '.join(hashtags)}" if base_caption else " ".join(hashtags)

            enriched_data: EnrichedHighlightData = {
                "start": start_time,
                "end": end_time,
                "segment_text": segment_text,
                "caption_with_hashtags": caption_with_hashtags,
                "title": title or None,
                "description": description or None,
                "hashtags": hashtags or None,
            }
            enriched_highlights.append(enriched_data)

        if not enriched_highlights:
            print("Не удалось обогатить ни один хайлайт метаданными.")
            return []

        print(f"\nУспешно обогащено хайлайтов: {len(enriched_highlights)}.")
        return enriched_highlights

    except Exception as e:
        print(f"Ошибка в GetHighlights: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    example_transcription = """
    [0.0] Speaker 1: Welcome to our discussion about artificial intelligence.
    [15.5] Speaker 1: Today we'll explore the fascinating world of machine learning.
    [30.2] Speaker 2: One of the most exciting applications is in video processing.
    [45.8] Speaker 1: Let's look at how AI can automatically generate video highlights.
    [60.0] Speaker 2: This technology is revolutionizing content creation, making it faster and easier.
    [75.5] Speaker 1: We're seeing it used widely in social media, entertainment, and even education platforms to deliver personalized content.
    [90.2] Speaker 2: The ability for AI to not just cut clips but understand context and find truly engaging moments is key.
    [105.8] Speaker 1: It's changing how we create and consume digital content daily. Think about personalized news feeds.
    [120.0] Speaker 2: Let's dive into some specific examples of tools available now.
    [135.0] Speaker 1: Good idea. Tool number one uses advanced natural language processing...
    """

    final_highlights = GetHighlights(example_transcription)
    if final_highlights:
        print("\n--- Итоговые обогащённые хайлайты ---")
        for i, highlight in enumerate(final_highlights, 1):
            print(f"Хайлайт {i}:")
            print(f"  Время: {highlight['start']:.2f}s - {highlight['end']:.2f}s")
            print(f"  Текст: {highlight['segment_text'][:100]}...") # Фрагмент текста
            print(f"  Подпись: {highlight['caption_with_hashtags']}")
            print("-" * 10)
    else:
        print("\nНе найдено валидных обогащённых хайлайтов.")

# --- Utility: tone/keywords heuristic ---
def compute_tone_and_keywords(text: str) -> dict:
    """
    Простая эвристика для определения «тона» фрагмента и набора ключевых слов.
    Возвращает:
        {
          "tone": "urgency" | "drama" | "positive" | "neutral",
          "keywords": ["слово1", "слово2", ...]  # нижний регистр, без пунктуации
        }
    Правила:
      - Подсчёт вхождений индикаторов по категориям + бонус за '!' для интенсивности.
      - При равенстве/отсутствии индикаторов — tone="neutral".
      - Ключевые слова: 1–4 "жёлуди" (нижний регистр), из найденных индикаторных слов,
        приоритизация по частоте и длине, со стоп-словами.
    """
    try:
        if not text or not str(text).strip():
            return {"tone": "neutral", "keywords": []}

        raw = str(text)
        s = raw.lower()

        # Индикаторы (стеммы/подстроки допустимы)
        indicators = {
            "urgency": {"срочно", "быстро", "скорее", "внимание", "опасн", "атак", "пожар", "беги"},
            "drama": {"плач", "потеря", "больно", "умер", "траг", "слёзы", "страх", "кровь"},
            "positive": {"круто", "супер", "ура", "рад", "люблю", "счаст", "класс", "смешно"},
        }
        emoji_to_tone = {"🔥": "urgency", "❤️": "positive", "😂": "positive", "💀": "drama"}

        # Подсчёт вхождений индикаторов (подстрочные совпадения, чтобы ловить стеммы)
        scores = {"urgency": 0, "drama": 0, "positive": 0}
        for tone, tokens in indicators.items():
            sc = 0
            for tok in tokens:
                try:
                    sc += s.count(tok)  # подстрочные совпадения
                except Exception:
                    continue
            scores[tone] += sc

        # Эмодзи-влияние
        for emo, tone_name in emoji_to_tone.items():
            if emo in raw:
                scores[tone_name] += raw.count(emo)

        # Бонус за '!' как прокси интенсивности -> скорее в сторону urgency
        exclam = raw.count("!")
        if exclam > 0:
            scores["urgency"] += exclam

        # Выбор тона по максимуму; при равенстве или нуле — neutral
        best_tone = "neutral"
        try:
            max_val = max(scores.values()) if scores else 0
            if max_val > 0:
                # если несколько с одинаковым максимумом — neutral
                winners = [k for k, v in scores.items() if v == max_val]
                best_tone = winners[0] if len(winners) == 1 else "neutral"
        except Exception:
            best_tone = "neutral"

        # --- Извлечение ключевых слов ---
        # Токенизация: русские/латинские буквы и цифры, остальное — разделители
        import re
        tokens = re.findall(r"[а-яёa-z0-9]+", s, flags=re.IGNORECASE)
        if not tokens:
            return {"tone": best_tone, "keywords": []}

        stop = {"и", "в", "на", "это", "как", "что", "я", "мы", "они", "он", "она", "а", "но", "или", "да"}
        # Частоты
        freq: dict[str, int] = {}
        for t in tokens:
            if t in stop:
                continue
            freq[t] = freq.get(t, 0) + 1

        # Кандидаты — те токены, которые соприкасаются с любым индикаторным стеммом
        indicator_stems = set().union(*indicators.values())
        candidates = []
        for word, cnt in freq.items():
            if any(stem in word for stem in indicator_stems):
                candidates.append((word, cnt, len(word)))

        # Сортировка по убыванию: частота, длина; затем лексикографически
        candidates.sort(key=lambda x: (-x[1], -x[2], x[0]))

        # Выбрать до 4 уникальных по слову
        kw = []
        seen = set()
        for w, _, _ in candidates:
            if w not in seen:
                seen.add(w)
                kw.append(w)
            if len(kw) >= 4:
                break

        return {"tone": best_tone, "keywords": kw}
    except Exception:
        # На всякий случай — фолбэк: полный бэкомпат
        return {"tone": "neutral", "keywords": []}

# --- Utility: emoji heuristics ---
def compute_emojis_for_segment(text: str, tone: str, max_count: int) -> list[str]:
    """
    emoji: heuristics and placement
    Автономная эвристика выбора эмодзи на основе тона и ключевых слов.

    Правила:
    - Если max_count <= 0 -> [].
    - Карта по тону:
        urgency: ["🔥", "⚠️", "💥"]
        drama: ["💔", "😢", "💀"]
        positive: ["😂", "✨", "🎉", "😊"]
        neutral: []
    - Ключевые слова (рус. стеммы/подстроки):
        смех/юмор -> приоритет "😂" | триггеры: {"смешн","ха","лол","ахаха"}
        экшн/напряжение -> "🔥","⚠️" | триггеры: {"срочно","беги","атак","взрыв","пожар"}
        радость/успех -> "🎉","✨" | триггеры: {"ура","побед","круто","супер","класс"}
    - Сначала кандидаты по ключевым словам (в указанном порядке групп), затем добиваем из карты тона.
    - Удаляем дубликаты, обрезаем до max_count.
    - Никаких внешних вызовов, зависимостей — чистая локальная функция.
    """
    try:
        if max_count is None:
            max_count = 0
        try:
            max_count = int(max_count)
        except Exception:
            max_count = 0

        if max_count <= 0:
            return []

        s = str(text or "")
        s_lower = s.lower()

        tone_map = {
            "urgency": ["🔥", "⚠️", "💥"],
            "drama": ["💔", "😢", "💀"],
            "positive": ["😂", "✨", "🎉", "😊"],
            "neutral": [],
        }

        # Ключевые слова (стеммы/подстроки)
        kw_laughter = {"смешн", "ха", "лол", "ахаха"}
        kw_action   = {"срочно", "беги", "атак", "взрыв", "пожар"}
        kw_joy      = {"ура", "побед", "круто", "супер", "класс"}

        candidates: list[str] = []

        # Смех/юмор — добавляем "😂" один раз, если найдено совпадение
        if any(tok in s_lower for tok in kw_laughter):
            candidates.append("😂")

        # Экшн/напряжение — "🔥" и "⚠️" (в таком порядке)
        if any(tok in s_lower for tok in kw_action):
            candidates.extend(["🔥", "⚠️"])

        # Радость/успех — "🎉" и "✨"
        if any(tok in s_lower for tok in kw_joy):
            candidates.extend(["🎉", "✨"])

        # Добиваем из карты тона
        t = str(tone or "neutral").lower().strip()
        tone_candidates = tone_map.get(t, [])
        candidates.extend(tone_candidates)

        # Уникализация с сохранением порядка
        seen = set()
        result: list[str] = []
        for emo in candidates:
            if emo not in seen:
                seen.add(emo)
                result.append(emo)
            if len(result) >= max_count:
                break

        return result[:max_count]
    except Exception:
        # На всякий случай — "тихий" фолбэк
        return []
