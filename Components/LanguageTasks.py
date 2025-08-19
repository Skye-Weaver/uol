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
Ты — креатор коротких видео для соцсетей. По предоставленной транскрипции выдели как можно больше непересекающихся отрезков, которые подойдут для увлекательных коротких роликов. Отдавай приоритет разнообразию валидных сегментов.
Верни ТОЛЬКО JSON-массив объектов. Каждый объект обязан содержать ключи "start" и "end" (на английском) с точными временными метками начала и конца сегмента из транскрипта. Никакого текста, объяснений или форматирования вне JSON.

Критерии выбора:
• Ключевые мысли, объяснения, вопросы, выводы и вообще наиболее «цепляющие» места.
• Старайся включать завершённые мысли/предложения.
• Предпочтительны моменты с понятным контекстом, но при необходимости соблюдай ограничение по времени.
• Естественные паузы и переходы — плюс, но не обязательны.

Требования к длительности:
• Длительность каждого сегмента (end - start) СТРОГО ОТ {cfg.llm.highlight_min_sec} ДО {cfg.llm.highlight_max_sec} секунд (включительно).
• Сегменты не должны перекрываться.
• Найди и верни от 10 до {cfg.llm.max_highlights} валидных сегментов — не больше.

Точность таймкодов:
• Используй ИМЕННО те таймкоды, что присутствуют/логически вытекают из транскрипта.
• Нельзя придумывать или править таймкоды.

Пример JSON-вывода:
[
  {{"start": "8.96", "end": "42.20"}},
  {{"start": "115.08", "end": "156.12"}},
  {{"start": "1381.68", "end": "1427.40"}}
]

• Главная цель — найти несколько сегментов со сроком строго {cfg.llm.highlight_min_sec}–{cfg.llm.highlight_max_sec} секунд.
• Убедись, что таймкоды соответствуют фактическим маркерам/границам смысла.
• Сегменты не должны пересекаться.
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

            # Filter the highlights: Keep only those passing individual validation
            valid_highlights = [h for h in raw_highlights if validate_highlight(h)]

            if not valid_highlights:
                print("No highlights passed individual duration/format validation in this attempt.")
                continue # Try next attempt

            # Check for overlaps ONLY among the valid duration highlights
            # Sort again just to be sure, although validate_highlights also sorts internally for its check
            sorted_highlights = sorted(valid_highlights, key=lambda x: float(x["start"]))
            overlaps_found = False
            for i in range(len(sorted_highlights) - 1):
                if float(sorted_highlights[i]["end"]) > float(sorted_highlights[i + 1]["start"]):
                    print(f"Обнаружено пересечение между {sorted_highlights[i]} и {sorted_highlights[i+1]}")
                    overlaps_found = True
                    break # No need to check further overlaps in this attempt

            if overlaps_found:
                print("Проверка на пересечения провалена для этой попытки.")
                continue # Try next attempt

            # If we reach here, we have a non-empty list of valid, non-overlapping highlights
            print(f"Успех на попытке {attempt + 1}. Найдено валидных сегментов: {len(sorted_highlights)}.")
            # Apply max_highlights cap from config
            try:
                max_h = int(cfg.llm.max_highlights)
                if max_h > 0:
                    return sorted_highlights[:max_h]
            except Exception:
                pass
            return sorted_highlights # Return the validated and sorted list

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
BATCH_METADATA_SYSTEM_PROMPT = """
ТЫ ДОЛЖЕН:
1. Сгенерировать валидный JSON-массив метаданных для коротких видео (Shorts) на основе входного массива объектов с `id`.
2. Для каждого объекта сформировать:
   - title — Заголовок должен содержать **информативный хук**. Это может быть сформулировано как неожиданный вывод, полезный совет или вопрос, который заставляет задуматься целевую аудиторию. Избегай дешевого кликбейта, общих фраз ("Смотри до конца!") и вопросов с очевидным ответом.
   - description — до 150 символов, кратко раскрывает суть или основной тезис видео.
   - hashtags — массив из 3–5 строк, первый ВСЕГДА `#shorts`, остальные — релевантны теме.
КАК ЭТО ДЕЛАТЬ:
- Заголовок должен быть информативным, фокусируясь на основной идее или результате, избегать кликбейтных вопросов и пустых интриг.
- Описание — кратко объясняет, о чём видео или его ценность.
- Хэштеги — подбирать по тематике и технологии, первый — всегда #shorts.

    ПРАВИЛА:
1. Твой ответ должен быть ИСКЛЮЧИТЕЛЬНО одним валидным JSON-массивом.
2. Для каждого входного объекта с `id` ты должен сгенерировать соответствующий объект в выходном массиве.
3. Каждый выходной объект должен содержать три поля: `title`, `description`, `hashtags`.

---
Пример 1 (Техническая тема):
Вход: [{"id":"seg_1","text":"...обсудим, как `async/await` в Python не всегда ускоряет код, а при неправильном использовании сетевых запросов может его замедлить..."}]
Выход: [{"id":"seg_1","title":"Почему `async/await` может замедлить ваш Python-код?","description":"Разбираем частую ошибку при работе с асинхронностью, которая приводит к падению производительности.","hashtags":["#shorts","#python","#asyncio","#разработка"]}]

Пример 2 (Общая концепция):
Вход: [{"id":"seg_2","text":"...ключевая проблема ИИ-моделей — это предвзятость в обучающих данных, которая приводит к ошибкам. Многие разработчики забывают про очистку данных..."}]
Выход: [{"id":"seg_2","title":"Главная ошибка при работе с ИИ, которую все допускают","description":"Почему очистка данных важнее, чем выбор архитектуры модели? Краткий разбор проблемы предвзятости (bias).","hashtags":["#shorts","#ИИ","#datascience","#машинноеобучение"]}]
---
"""

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
