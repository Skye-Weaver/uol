import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ЛОКАЛЬНЫЙ СМОУК-ТЕСТ Gemini + Whisper (RU)
# Требования:
# 1) В корне проекта файл .env с GOOGLE_API_KEY=...
# 2) Наличие бинаря ffmpeg в PATH (в Colab есть по умолчанию)
# 3) Для теста ASR положите короткий аудиофайл "audio.wav" в корень проекта (или укажите путь ниже)
#
# Запуск:
#   python -m tests.smoke_gemini_and_whisper
#
# Что проверяет:
# - Инициация LLM-конвейера (Gemini 2.5 Flash) через Components/LanguageTasks.py
# - Генерация хайлайтов и описания/хэштегов (строгий JSON)
# - Инициация ASR (faster-whisper, RU), сегментная и словная транскрибация при наличии файла audio.wav

# Добавим корень проекта в sys.path для корректного импорта локальных модулей
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

def _print_rule(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def test_llm_pipeline():
    from Components import LanguageTasks  # импорт всего модуля, чтобы использовать его конфиг и клиент
    _print_rule("ТЕСТ LLM (Gemini 2.5 Flash, RU промпты, строгий JSON)")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("SKIP: GOOGLE_API_KEY не найден в окружении, пропускаю LLM-тест.")
        return

    # Минимальная русская транскрипция (формат как в main: [start] Speaker: text [end])
    sample_transcription = (
        "[0.00] Speaker: Привет! Сегодня обсудим, как автоматически находить лучшие моменты в видео. [12.50]\n"
        "[12.50] Speaker: Мы используем распознавание речи и языковые модели, чтобы выделять интересные фрагменты. [28.60]\n"
        "[30.00] Speaker: Задача — получить короткие клипы с ясной мыслью за 30–60 секунд. [45.50]\n"
        "[60.00] Speaker: Затем добавляем описание и хэштеги, чтобы повысить вовлечённость. [75.20]\n"
    )

    try:
        # Вызов основной функции конвейера
        highlights = LanguageTasks.GetHighlights(sample_transcription)
        if not highlights:
            print("LLM: Не получено хайлайтов (возможно лимиты/ключ/сетевые ограничения).")
            return

        print(f"LLM: Получено хайлайтов: {len(highlights)}. Пример первого:")
        first = highlights[0]
        # Ключи должны быть на английском согласно решению
        for k in ("start", "end", "caption_with_hashtags"):
            print(f"  {k}: {first.get(k)}")
    except Exception as e:
        print("LLM: Ошибка при генерации хайлайтов/описаний:", e)

def test_asr_pipeline(audio_path: str = "audio.wav"):
    from Components.Transcription import transcribeAudio, transcribe_segment_word_level
    _print_rule("ТЕСТ ASR (faster-whisper, RU) — сегменты и словные таймкоды")

    audio_file = Path(ROOT, audio_path)
    if not audio_file.exists():
        print(f"SKIP: аудиофайл для теста не найден: {audio_file}. Поместите 'audio.wav' в корень проекта.")
        return

    try:
        segs = transcribeAudio(str(audio_file))
        if segs:
            print(f"ASR: Сегментов: {len(segs)}. Пример первых 3:")
            for row in segs[:3]:
                # row = [text, start, end]
                print(f"  [{row[1]:.2f} - {row[2]:.2f}] {row[0][:80]}")
        else:
            print("ASR: Сегментная транскрибация вернула пустой результат.")

        word_level = transcribe_segment_word_level(str(audio_file))
        if word_level and word_level.get("segments"):
            print(f"ASR: Сегментов (слова): {len(word_level['segments'])}. Пример первых слов первого сегмента:")
            first_words = word_level["segments"][0].get("words", [])
            for w in first_words[:10]:
                print(f"  {w.get('start', 0):.2f} - {w.get('end', 0):.2f}: {w.get('text','')}")
        else:
            print("ASR: Словная транскрибация отсутствует или пустая.")
    except Exception as e:
        print("ASR: Ошибка при транскрибировании:", e)

def main():
    _print_rule("НАЧАЛО СМОУК-ТЕСТОВ")
    # LLM
    test_llm_pipeline()
    # ASR
    test_asr_pipeline("audio.wav")
    _print_rule("КОНЕЦ СМОУК-ТЕСТОВ")

if __name__ == "__main__":
    main()