# AI-Youtube-Shorts-Generator-Gemini

An AI-powered tool that automatically generates engaging short-form videos from longer YouTube content, optimized for platforms like YouTube Shorts, Instagram Reels, and TikTok and for static videos with a 1 person speaking.

## Key Features

- **Smart Video Download**: 
  - Downloads videos from YouTube URLs with quality selection
  - Supports both progressive and adaptive streams
  - Automatically merges video and audio for best quality
  - Handles local video files as input

- **Advanced Transcription**:
  - Uses `faster-whisper` (base.en model) for efficient transcription
  - Provides both segment-level and word-level timestamps
  - CPU-optimized processing with int8 quantization
  - Multi-threaded performance for faster processing

- **AI-Powered Highlight Detection**:
  - Leverages Google's Gemini-2.0-flash model for content analysis
  - Identifies the most engaging segments from transcriptions
  - Generates relevant hashtags and captions
  - Smart content selection based on engagement potential

- **Intelligent Video Processing**:
  - Multiple vertical cropping strategies:
    - Static centered crop
    - Face-detection based dynamic cropping
    - Average face position based cropping
  - Maintains optimal 9:16 aspect ratio for shorts
  - Automatic bottom margin cropping for better framing
  - Supports both static and animated captions

- **Robust Caching System**:
  - SQLite database for efficient data management
  - Caches processed videos, audio, and transcriptions
  - Prevents redundant processing of previously handled content
  - Easy cache management and cleanup

## Prerequisites

- Python 3.10 or higher
- FFmpeg (latest version recommended)
- CUDA-compatible GPU (optional, for faster processing)
- 4GB+ RAM recommended

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Youtube-Shorts-Generator.git
   cd AI-Youtube-Shorts-Generator
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your_google_ai_studio_key_here
   ```

## Usage

1. Start the tool:
   ```bash
   python main.py
   ```

2. Input either:
   - A YouTube URL
   - A path to a local video file

3. Select video quality when prompted (for YouTube downloads)

4. The tool will process your video through several stages:
   - Download/import video
   - Extract and transcribe audio
   - Identify engaging segments
   - Create vertical crops
   - Add captions
   - Generate final shorts

5. Find your processed shorts in the `shorts` directory

## Configuration Options

- `USE_ANIMATED_CAPTIONS`: Toggle between static and animated captions (in main.py) (reccomended)
- `SHORTS_DIR`: Customize output directory for processed videos
- CPU thread optimization in `Components/Transcription.py`

## Project Structure

```
AI-Youtube-Shorts-Generator/
├── Components/
│   ├── Captions.py       # Caption generation and rendering
│   ├── Database.py       # SQLite database management
│   ├── Edit.py          # Video editing and processing
│   ├── FaceCrop.py      # Vertical cropping algorithms
│   ├── LanguageTasks.py # AI content analysis
│   ├── Speaker.py       # Speaker detection (experimental)
│   ├── Transcription.py # Audio transcription
│   └── YoutubeDownloader.py # Video download handling
├── main.py              # Main execution script
├── requirements.txt     # Python dependencies
└── .env                # Environment variables
```

## Database Schema

The SQLite database (`video_processing.db`) contains three main tables:

1. **videos**:
   - id (PRIMARY KEY)
   - youtube_url
   - local_path
   - audio_path
   - created_at

2. **transcriptions**:
   - id (PRIMARY KEY)
   - video_id (FOREIGN KEY)
   - transcription_data
   - created_at

3. **highlights**:
   - id (PRIMARY KEY)
   - video_id (FOREIGN KEY)
   - start_time
   - end_time
   - output_path
   - segment_text
   - caption_with_hashtags
   - created_at

## Known Issues & Limitations

1. **Face Detection**:
   - The face-based cropping can be inconsistent with multiple faces
   - May need manual adjustment for optimal framing in some cases

2. **Speaker Detection**:
   - Current implementation uses basic voice activity detection
   - Full speaker diarization not yet implemented

3. **Resource Usage**:
   - Processing long videos can be memory-intensive
   - GPU acceleration limited to specific components

## Troubleshooting

1. If facing cache-related issues:
   - Delete `video_processing.db` to clear the cache
   - Remove temporary files in the `videos` directory

2. For video processing errors:
   - Ensure FFmpeg is properly installed and accessible
   - Check available disk space for temporary files
   - Verify input video format compatibility

3. For AI-related issues:
   - Confirm Google API key is valid and has sufficient quota
   - Check internet connectivity for API calls

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- SQL integration made by [YassineKADER](https://github.com/YassineKADER/AI-Youtube-Shorts-Generator-)
- Original project by [SamurAIGPT](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator)
- Uses Google's Gemini AI for content analysis
- Powered by faster-whisper for transcription

## Batch‑метаданные

- Цель: пакетно сгенерировать для каждого текстового фрагмента видео метаданные: `title`, `description`, `hashtags`.
- Вход: JSON‑массив объектов вида `{"id": string, "text": string}`.
- Выход: JSON‑массив объектов вида `{"id": string, "title": string, "description": string, "hashtags": string[]}`.
  - Требования:
    - title: 40–70 символов
    - description: ≤ 150 символов
    - hashtags: 3–5 шт., первый элемент строго `#shorts`
    - строго один валидный JSON‑массив без текста вокруг
- Реализация: [generate_metadata_batch()](Components/LanguageTasks.py:576)
- Место интеграции в пайплайн: [GetHighlights()](Components/LanguageTasks.py:711) — после выделения тайм‑сегментов и извлечения текста для каждого сегмента.

Новый системный промпт (точно как в ТЗ), применяемый при пакетной генерации:
```
Ты — эксперт по SMM и продвижению на YouTube, специализирующийся на вирусных Shorts. Тебе на вход подается JSON-массив текстовых фрагментов из видео, каждый с уникальным `id`. Твоя задача — для каждого фрагмента создать оптимальный набор метаданных для максимального вовлечения и охвата.

Правила:
1. Твой ответ должен быть ИСКЛЮЧИТЕЛЬНО одним валидным JSON-массивом. Никакого текста до или после.
2. Для каждого входного объекта с `id` ты должен сгенерировать объект в выходном массиве с тем же `id` и тремя полями: `title`, `description` и `hashtags`.
3. title (заголовок): 40–70 символов, интригующий, задает вопрос или создает предвкушение. Обязательно использовать ключевые слова из текста.
4. description (описание): до 150 символов, кратко раскрывает суть, допускается призыв к действию.
5. hashtags (хэштеги): массив из 3–5 строк; первым ВСЕГДА `#shorts`; остальные — максимально релевантны теме фрагмента.

Пример Входа:
[{"id":"seg_1","text":"Сегодня обсудим, как автоматически находить лучшие моменты в видео..."}]

Пример Выхода:
[{"id":"seg_1","title":"Нейросеть находит лучшие моменты в видео?","description":"Смотрите, как ИИ анализирует ролики для создания шортсов.","hashtags":["#shorts","#ИИ","#нейросети","#видеомонтаж"]}]
```

## Rate‑limiting

- Централизованная обёртка для вызова LLM: [call_llm_with_retry()](Components/LanguageTasks.py:145)
  - Пытается повторить запрос при лимитах API и парсит задержку повтора.
  - Разбор задержки: [parse_retry_delay_seconds()](Components/LanguageTasks.py:89)
- Поддерживаемые форматы retryDelay:
  - `Retry-After: 28`
  - `retry-after: 28`
  - `"retryDelay": "28s"`
  - `retryDelay: 28s`
- Логи (точные формулировки):
  - "Лимит API обработан. Выполняю паузу на X секунд перед попыткой #Y."
  - "Не удалось извлечь retryDelay. Попытки прекращены."
- Поведение:
  - При наличии корректного `retryDelay` выполняется `sleep(X)` и повтор запроса.
  - При отсутствии `retryDelay` попытки прекращаются и исключение пробрасывается дальше.

## Конфигурация (config.yaml)

- Путь: [config.yaml](config.yaml)
- Структура секций и ключи:
  - `processing`:
    - `use_animated_captions`: bool — использовать ли анимированные субтитры
    - `shorts_dir`: str — каталог для итоговых шортов
    - `videos_dir`: str — каталог для промежуточных файлов/видео
    - `crop_bottom_percent`: float — нижний кроп вертикального видео (в процентах)
    - `min_video_dimension_px`: int — минимальный размер видео
    - `log_transcription_preview_len`: int — длина превью транскрипции в логах
  - `llm`:
    - `model_name`: str — модель Gemini
    - `temperature_highlights`: float — температура для поиска хайлайтов
    - `temperature_metadata`: float — температура для метаданных
    - `max_attempts_highlights`: int — попытки при извлечении сегментов
    - `max_attempts_metadata`: int — попытки при генерации метаданных
    - `highlight_min_sec` / `highlight_max_sec`: границы длительности сегмента
    - `max_highlights`: максимум сегментов
- Минимальный пример:
```yaml
processing:
  use_animated_captions: true
  shorts_dir: "shorts"
  videos_dir: "videos"

llm:
  model_name: "gemini-2.5-flash"
  temperature_highlights: 0.2
  temperature_metadata: 1.0
  highlight_min_sec: 29
  highlight_max_sec: 61
  max_highlights: 20
```
- Поведение при отсутствии файла: используются значения по умолчанию; при этом выводится сообщение
  "Конфиг не найден. Использую значения по умолчанию." — см. [Components/config.py](Components/config.py:100).
- Логирование факта загрузки и активной модели — см. [main.py](main.py:15).

## Централизованные пути и конфигурация

Начиная с актуальной версии, все ресурсы и каталоги резолвятся относительно базовой директории из конфига. Ключевые параметры задаются в секции `paths` и `processing`.

Пример фрагмента `config.yaml` (минимальный):
```yaml
paths:
  base_dir: .
  fonts_dir: fonts
processing:
  transcriptions_dir: transcriptions
  shorts_dir: shorts
```

Назначение ключей:
- `base_dir` — корень проекта, относительно которого резолвятся внутренние пути.
- `fonts_dir` — подкаталог со шрифтами (по умолчанию `fonts`), резолвится относительно `base_dir`.
- `transcriptions_dir` — директория, куда сохраняются транскрипции (`.txt/.json/.srt/.vtt`).
- `shorts_dir` — директория, куда сохраняются итоговые шорт‑видео.

Центральные функции резолва путей:
- Ресурсы: [resolve_path()](Components/Paths.py:22)
- Шрифты: [fonts_path()](Components/Paths.py:37)

Пример резолва пути к шрифту «Montserrat-Bold.ttf» с настройками по умолчанию:
- Конфиг: `paths.base_dir: .`, `paths.fonts_dir: fonts`
- Вызов: [fonts_path()](Components/Paths.py:37) для `"Montserrat-Bold.ttf"` даст путь вида: `<ABS_BASE_DIR>/fonts/Montserrat-Bold.ttf`

## Именование выходных short‑файлов

Уникальные имена итоговых и временных short‑файлов формируются функцией [build_short_output_name()](Components/Paths.py:6).

Шаблоны:
- Итоговый: `shorts/{base_name}_highlight_{idx:02d}_final.mp4`  
  Пример: `shorts/master-ABC_highlight_01_final.mp4`
- Временный (анимация): `{final_path}_temp_anim.mp4`

Индекс `idx` — это порядковый номер хайлайта в текущей сессии; он пробрасывается из цикла и логируется (см. [main.py](main.py)).

## Экспорт транскрипций

После завершения унифицированной транскрипции ("Unified transcription complete…") экспорт выполняется автоматически, код — [Components/Transcription.py](Components/Transcription.py).

Создаются файлы:
- `{transcriptions_dir}/{base_name}.txt`
- `{transcriptions_dir}/{base_name}.json`
- `{transcriptions_dir}/{base_name}.srt`
- `{transcriptions_dir}/{base_name}.vtt`

Особенности:
- JSON сохраняет сегменты и слова в UTF‑8 (`ensure_ascii=False`).
- Папка назначения задаётся через `processing.transcriptions_dir` в `config.yaml`.

## Пути и совместимость окружений

- Абсолютные пути вида `/content/uol/*` больше не используются — все ресурсы резолвятся относительно `paths.base_dir`.
- Для Google Colab/контейнеров можно установить `paths.base_dir` на рабочую директорию окружения — остальные относительные пути (`fonts_dir`, `transcriptions_dir`, `shorts_dir`) подхватятся корректно.

### Релевантные тесты

- Проверка уникальности имён short‑файлов: [tests/test_output_naming.py](tests/test_output_naming.py)
- Экспорт транскрипции в 4 формата: [tests/test_transcription_export.py](tests/test_transcription_export.py)
- Резолв путей для ресурсов и шрифта: [tests/test_resource_paths.py](tests/test_resource_paths.py)
## Тесты

- Запуск всех тестов:
  - `python -m unittest -v`
- Покрытие:
  - Форматирование транскрипции [build_transcription_prompt()](Components/LanguageTasks.py:38) —
    [tests/test_language_tasks_prompt_formatting.py](tests/test_language_tasks_prompt_formatting.py)
  - Пакетная генерация метаданных [generate_metadata_batch()](Components/LanguageTasks.py:576) —
    [tests/test_language_tasks_batch_metadata.py](tests/test_language_tasks_batch_metadata.py)
  - Обработка лимитов API [call_llm_with_retry()](Components/LanguageTasks.py:145) —
    [tests/test_language_tasks_rate_limit.py](tests/test_language_tasks_rate_limit.py)
- Для запуска не требуется реальный API/ключ: в тестах используется monkeypatch/mock.
- Опциональный онлайновый сценарий (при наличии GOOGLE_API_KEY): [tests/smoke_gemini_and_whisper.py](tests/smoke_gemini_and_whisper.py)
