from google import genai
from typing import TypedDict, List, Optional
import json
import os
import re # Import regex for parsing transcription
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "Google API key not found. Make sure it is defined in the .env file."
    )

client = genai.Client(
        api_key=GOOGLE_API_KEY,
    )
# Consider using a more capable model if generating descriptions needs more nuance
# model = genai.GenerativeModel("gemini-1.5-flash") # Example alternative
model = "gemini-2.5-flash"

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


def validate_highlight(highlight: HighlightSegment) -> bool:
    """Validate a single highlight segment's time duration and format."""
    try:
        if not all(key in highlight for key in ["start", "end"]):
            print(f"Validation Fail: Missing 'start' or 'end' key in {highlight}")
            return False

        start = float(highlight["start"])
        end = float(highlight["end"])
        duration = end - start

        # Check for valid duration (between ~29 and ~61 seconds)
        min_duration = 30.0 - 1.0 # Increased tolerance
        max_duration = 60.0 + 1.0 # Increased tolerance

        if not (min_duration <= duration <= max_duration):
            print(f"Validation Fail: Duration {duration:.2f}s out of range [~29s, ~61s] for {highlight}")
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
    system_instruction_text = """
Ты — креатор коротких видео для соцсетей. По предоставленной транскрипции выдели как можно больше непересекающихся отрезков, которые подойдут для увлекательных коротких роликов. Отдавай приоритет разнообразию валидных сегментов.
Верни ТОЛЬКО JSON-массив объектов. Каждый объект обязан содержать ключи "start" и "end" (на английском) с точными временными метками начала и конца сегмента из транскрипта. Никакого текста, объяснений или форматирования вне JSON.

Критерии выбора:
• Ключевые мысли, объяснения, вопросы, выводы и вообще наиболее «цепляющие» места.
• Старайся включать завершённые мысли/предложения.
• Предпочтительны моменты с понятным контекстом, но при необходимости соблюдай ограничение по времени.
• Естественные паузы и переходы — плюс, но не обязательны.

Требования к длительности:
• Длительность каждого сегмента (end - start) СТРОГО ОТ 30 ДО 60 секунд (включительно).
• Сегменты не должны перекрываться.
• Найди и верни от 10 до 20 валидных сегментов — не больше.

Точность таймкодов:
• Используй ИМЕННО те таймкоды, что присутствуют/логически вытекают из транскрипта.
• Нельзя придумывать или править таймкоды.

Пример JSON-вывода:
[
  {"start": "8.96", "end": "42.20"},
  {"start": "115.08", "end": "156.12"},
  {"start": "1381.68", "end": "1427.40"}
]

• Главная цель — найти несколько сегментов со сроком строго 30–60 секунд.
• Убедись, что таймкоды соответствуют фактическим маркерам/границам смысла.
• Сегменты не должны пересекаться.
    """

    # Define generation config based on AI Studio code
    generation_config = make_generation_config(system_instruction_text, temperature=0.2)

    for attempt in range(max_attempts):
        print(f"\nПопытка {attempt + 1}: генерация и валидация тайм-сегментов для хайлайтов...")
        try:
            # Structure the prompt and system instruction for generate_content
            user_prompt_text = f"Transcription:\\n{transcription}"
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt_text)])]
            # Use the global model but with the new config
            response = client.models.generate_content(model=model,contents=contents,
                                              config=generation_config)

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
            return sorted_highlights # Return the validated and sorted list

        except json.JSONDecodeError:
             print(f"Неудача на попытке {attempt + 1}: некорректный JSON от LLM.")
             if 'response_text' in locals(): print(f"Сырой ответ LLM: {response_text}")
             continue
        except Exception as e:
            print(f"Неудача на попытке {attempt + 1}: непредвиденная ошибка: {str(e)}")
            if 'response_text' in locals(): print(f"Сырой ответ LLM при ошибке: {response_text}")
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
    generation_config = make_generation_config(system_prompt, temperature=1)

    for attempt in range(max_attempts):
        try:
            # Structure the prompt and system instruction for generate_content
            user_prompt_text = f"Segment Text:\n{segment_text}"
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt_text)])]
            
            # Use the global model with the new config
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generation_config
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
            print(f"Неудача на попытке {attempt + 1}: непредвиденная ошибка при генерации описания: {str(e)}")
            if 'response_text' in locals(): print(f"Сырой ответ LLM при ошибке: {response_text}")
            continue

    print("Достигнуто максимальное число попыток генерации описания/хэштегов. Возвращаю None.")
    return None


# --- Updated Main Function ---

def GetHighlights(transcription: str) -> List[EnrichedHighlightData]:
    """
    Main function to get multiple highlight segments from transcription,
    each enriched with an LLM-generated description and hashtags.
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

        print(f"\nПерехожу к генерации описаний для {len(highlight_segments)} сегментов...")

        # 2. For each segment, extract text and generate description/hashtags
        for segment in highlight_segments:
            # Convert string timestamps to floats first
            try:
                start_time = float(segment["start"])
                end_time = float(segment["end"])
            except ValueError:
                print(f"Предупреждение: не удалось преобразовать таймкоды в float для сегмента {segment}. Пропускаю.")
                continue

            # 2a. Extract text for this segment
            segment_text = extract_text_for_segment(transcription, start_time, end_time)

            if not segment_text.strip():
                 print("Предупреждение: для этого сегмента не извлечён текст. Пропускаю генерацию описания.")
                 # Optionally skip this segment entirely or add placeholder description
                 continue # Skip this segment

            # 2b. Generate description and hashtags for the segment text
            caption_string = generate_description_and_hashtags(segment_text)

            if caption_string:
                # 2c. Combine time segment with description data
                enriched_data: EnrichedHighlightData = {
                    "start": start_time,
                    "end": end_time,
                    "segment_text": segment_text, # Store the original text
                    "caption_with_hashtags": caption_string
                }
                enriched_highlights.append(enriched_data)
            else:
                print(f"Предупреждение: не удалось сгенерировать описание для сегмента {start_time:.2f}-{end_time:.2f}. Пропускаю.")
                # Optionally add segment with placeholder/error description instead of skipping

        if not enriched_highlights:
            print("Не удалось обогатить ни один хайлайт описанием.")
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
