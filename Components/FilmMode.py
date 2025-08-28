"""
Режим "фильм" для анализа видео и выделения лучших моментов.
Анализирует длинные видео и предлагает оптимальные фрагменты для создания фильма из лучших частей.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import os
from datetime import datetime

from Components.YoutubeDownloader import download_youtube_video
from Components.Transcription import transcribe_unified
from Components.LanguageTasks import (
    build_transcription_prompt,
    GetHighlights,
    call_llm_with_retry,
    make_generation_config
)
from Components.Database import VideoDatabase
from Components.config import get_config, AppConfig
from Components.Logger import logger
from faster_whisper import WhisperModel


@dataclass
class FilmMoment:
    """Структура для хранения информации о моменте фильма"""
    moment_type: str  # "COMBO" или "SINGLE"
    start_time: float
    end_time: float
    text: str
    segments: List[Dict[str, Any]] = field(default_factory=list)  # Для COMBO: суб-сегменты
    context: str = ""  # Описание контекста


@dataclass
class RankedMoment:
    """Ранжированный момент с оценками"""
    moment: FilmMoment
    scores: Dict[str, float]  # Оценки по критериям
    total_score: float
    rank: int


@dataclass
class FilmAnalysisResult:
    """Результат анализа фильма"""
    video_id: str
    duration: float
    keep_ranges: List[Dict[str, Any]]
    scores: List[Dict[str, Any]]
    preview_text: str
    risks: List[str]
    metadata: Dict[str, Any]


class FilmAnalyzer:
    """
    Анализатор фильмов для выделения лучших моментов.
    Интегрируется с существующими компонентами проекта.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.db = VideoDatabase()

        # Настройки для режима фильм
        self.film_config = config.film_mode

    def analyze_film(self, url: Optional[str] = None, local_path: Optional[str] = None) -> FilmAnalysisResult:
        """
        Основной пайплайн анализа фильма
        1. Получение видео
        2. Транскрибация
        3. Анализ моментов через ИИ
        4. Ранжирование
        5. Обрезка скучных секунд
        6. Формирование результата
        """
        logger.logger.info("Начало анализа фильма в режиме 'фильм'")

        # 1. Получение видео и транскрибация
        video_path, transcription_data = self._get_video_and_transcription(url, local_path)
        if not video_path or not transcription_data:
            raise ValueError("Не удалось получить видео или транскрибацию")

        # Логируем информацию о полученных данных
        duration = transcription_data.get('duration', 0)
        segments_count = len(transcription_data.get('segments', []))
        logger.logger.info(f"Получены данные транскрибации: длительность={duration:.2f}s, сегментов={segments_count}")

        # 2. Анализ моментов через ИИ
        moments = self._analyze_moments(transcription_data)
        if not moments:
            logger.logger.warning("Не найдено подходящих моментов для анализа")
            logger.logger.warning(f"Данные транскрибации: duration={duration}, segments={segments_count}")
            return self._create_empty_result(video_path)

        # 3. Ранжирование моментов
        ranked_moments = self._rank_moments(moments)

        # 4. Обрезка скучных секунд
        trimmed_moments = self._trim_boring_segments(ranked_moments, transcription_data)

        # 5. Формирование результата
        result = self._create_result(video_path, trimmed_moments, transcription_data)

        logger.logger.info(f"Анализ фильма завершен. Найдено {len(trimmed_moments)} моментов")
        return result

    def _get_video_and_transcription(self, url: Optional[str], local_path: Optional[str]) -> tuple:
        """Получение видео и его транскрибация"""
        try:
            # Получение видео
            if url:
                logger.logger.info(f"Загрузка видео по URL: {url}")
                video_path = download_youtube_video(url)
            elif local_path:
                logger.logger.info(f"Использование локального файла: {local_path}")
                video_path = local_path
            else:
                raise ValueError("Не указан URL или локальный путь к видео")

            if not video_path or not os.path.exists(video_path):
                raise FileNotFoundError(f"Видео файл не найден: {video_path}")

            # Транскрибация
            logger.logger.info("Начало транскрибации видео")
            model = self._load_whisper_model()
            segments_legacy, word_level_transcription = transcribe_unified(video_path, model)

            # Используем длительность из модели Whisper вместо ffprobe
            duration = 0.0
            if word_level_transcription and 'segments' in word_level_transcription:
                # Извлекаем длительность из последнего сегмента
                segments = word_level_transcription['segments']
                if segments:
                    last_segment = segments[-1]
                    duration = float(last_segment.get('end', 0.0))

            # Если длительность все еще 0, пробуем ffprobe как fallback
            if duration == 0.0:
                try:
                    duration = self._get_video_duration(video_path)
                    logger.logger.info(f"Длительность получена через ffprobe: {duration:.2f} секунд")
                except Exception as e:
                    logger.logger.warning(f"Не удалось получить длительность через ffprobe: {e}")

            logger.logger.info(f"Финальная длительность видео: {duration:.2f} секунд")

            transcription_data = {
                'segments': segments_legacy,
                'word_level': word_level_transcription,
                'duration': duration
            }

            return video_path, transcription_data

        except Exception as e:
            logger.logger.error(f"Ошибка при получении видео или транскрибации: {e}")
            return None, None

    def _load_whisper_model(self) -> WhisperModel:
        """Загрузка модели Whisper"""
        # Используем ту же логику, что и в main.py
        from Components.config import reload_config
        cfg = reload_config()

        # Определение параметров модели
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except:
            has_cuda = False

        if has_cuda and cfg.logging.gpu_priority_mode:
            device = "cuda"
            model_size = "large-v3"
            compute_type = "float16"
        else:
            device = "cpu"
            model_size = "small"
            compute_type = "int8"

        logger.logger.info(f"Загрузка модели Whisper: {model_size} на {device}")
        return WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            num_workers=2,
        )

    def _get_video_duration(self, video_path: str) -> float:
        """Получение длительности видео через ffprobe"""
        try:
            import subprocess
            import json

            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data['format']['duration'])
            else:
                logger.logger.warning("Не удалось получить длительность видео через ffprobe")
                return 0.0
        except Exception as e:
            logger.logger.warning(f"Ошибка при получении длительности видео: {e}")
            return 0.0

    def _analyze_moments(self, transcription_data: Dict[str, Any]) -> List[FilmMoment]:
        """Анализ моментов через ИИ"""
        try:
            # Преобразование формата сегментов для build_transcription_prompt
            segments_legacy = transcription_data.get('segments', [])
            segments_dict = []

            # Преобразуем список списков в список словарей
            for seg in segments_legacy:
                if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                    segments_dict.append({
                        'text': str(seg[0]),
                        'start': float(seg[1]),
                        'end': float(seg[2])
                    })
                elif isinstance(seg, dict):
                    segments_dict.append(seg)

            # Формирование текста транскрибации
            transcription_text = build_transcription_prompt(segments_dict)

            # Анализ через LLM
            moments = self._extract_film_moments(transcription_text)

            logger.logger.info(f"Найдено {len(moments)} потенциальных моментов")
            return moments

        except Exception as e:
            logger.logger.error(f"Ошибка при анализе моментов: {e}")
            return []

    def _extract_film_moments(self, transcription: str) -> List[FilmMoment]:
        """Извлечение моментов фильма через LLM"""
        system_instruction = f"""
        Ты — эксперт по анализу видео контента для создания вирусных shorts. Проанализируй предоставленную транскрибацию и выдели лучшие моменты двух типов:

        1. COMBO (10-20 сек): Склейка 2-4 коротких кусков из одной сцены в хронологическом порядке для создания мини-дуги
        2. SINGLE (30-60 сек): Один самодостаточный момент с микро-аркой (завязка → нарастание → развязка)

        Критерии качества для shorts:
        - Эмоциональные пики и переломы статуса (признания, угрозы, ультиматумы, резкие смены намерения)
        - Конфликт и эскалация (столкновения, отрицания, оскорбления)
        - Панчлайны и остроумие (связка сетап → поворот → панч, сарказм, самоирония)
        - Цитатность/мемность (запоминающиеся фразы, афоризмы, каламбуры)
        - Ставки и цель (если X, то Y, последний шанс)
        - Крючки/клиффхэнгеры (вопросы, недосказанность)

        Верни ТОЛЬКО JSON-массив объектов с полями:
        - moment_type: "COMBO" или "SINGLE"
        - start_time: число (секунды)
        - end_time: число (секунды)
        - text: текст момента
        - context: краткое описание почему этот момент подходит

        Для COMBO также добавь:
        - segments: массив суб-сегментов с start/end/text

        Найди максимум {self.film_config.max_moments} лучших моментов.
        """

        try:
            logger.logger.info(f"Отправка запроса к LLM для анализа моментов (модель: {self.film_config.llm_model})")
            logger.logger.debug(f"Длина транскрибации: {len(transcription)} символов")

            generation_config = make_generation_config(system_instruction, temperature=0.3)

            response = call_llm_with_retry(
                system_instruction=None,
                content=transcription,
                generation_config=generation_config,
                model=self.film_config.llm_model,
            )

            if not response or not response.text:
                logger.logger.warning("LLM не вернул ответ при анализе моментов")
                return []

            # Парсинг JSON ответа
            response_text = response.text.strip()
            logger.logger.debug(f"Сырой ответ LLM: {response_text[:500]}...")

            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            try:
                moments_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.logger.error(f"Ошибка парсинга JSON от LLM: {e}")
                logger.logger.error(f"Текст для парсинга: {response_text[:200]}...")
                return []

            if not isinstance(moments_data, list):
                logger.logger.error(f"LLM вернул не массив, а {type(moments_data)}")
                return []

            moments = []
            for i, item in enumerate(moments_data):
                try:
                    if not isinstance(item, dict):
                        logger.logger.warning(f"Элемент {i} не является словарем, пропускаю")
                        continue

                    moment = FilmMoment(
                        moment_type=item.get('moment_type', 'SINGLE'),
                        start_time=float(item.get('start_time', 0)),
                        end_time=float(item.get('end_time', 0)),
                        text=item.get('text', ''),
                        context=item.get('context', ''),
                        segments=item.get('segments', [])
                    )
                    moments.append(moment)
                    logger.logger.debug(f"Обработан момент {i+1}: {moment.moment_type} {moment.start_time:.1f}-{moment.end_time:.1f}")

                except Exception as e:
                    logger.logger.warning(f"Ошибка при обработке момента {i}: {e}")
                    continue

            logger.logger.info(f"Успешно извлечено {len(moments)} моментов из {len(moments_data)}")
            return moments

        except Exception as e:
            logger.logger.error(f"Ошибка при извлечении моментов через LLM: {e}")
            import traceback
            logger.logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _rank_moments(self, moments: List[FilmMoment]) -> List[RankedMoment]:
        """Ранжирование моментов по баллам"""
        ranked_moments = []

        for i, moment in enumerate(moments):
            scores = self._calculate_moment_scores(moment)
            total_score = sum(
                score * self.film_config.ranking_weights.get(score_name, 0)
                for score_name, score in scores.items()
            )

            ranked_moment = RankedMoment(
                moment=moment,
                scores=scores,
                total_score=total_score,
                rank=0  # Будет установлено после сортировки
            )
            ranked_moments.append(ranked_moment)

        # Сортировка по убыванию общего балла
        ranked_moments.sort(key=lambda x: x.total_score, reverse=True)

        # Установка рангов
        for i, rm in enumerate(ranked_moments):
            rm.rank = i + 1

        logger.logger.info(f"Ранжирование завершено. Лучший момент имеет балл {ranked_moments[0].total_score:.2f}" if ranked_moments else "Нет моментов для ранжирования")

        return ranked_moments

    def _calculate_moment_scores(self, moment: FilmMoment) -> Dict[str, float]:
        """Расчет оценок момента по критериям"""
        scores = {}

        text = moment.text.lower()

        # Эмоциональные пики и переломы статуса
        emotional_keywords = ['признание', 'угроза', 'ультиматум', 'увольнение', 'я твой отец', 'ухожу', 'мы всё теряем', 'это был он']
        scores['emotional_peaks'] = sum(1 for keyword in emotional_keywords if keyword in text) * 2.0

        # Конфликт и эскалация
        conflict_keywords = ['нет', 'никогда', 'почему', 'хватит', 'оскорбление', 'жесткий', 'отрицание']
        scores['conflict_escalation'] = sum(1 for keyword in conflict_keywords if keyword in text) * 1.8

        # Панчлайны и остроумие
        wit_keywords = ['сарказм', 'самоирония', 'панчлайн', 'остроумие', 'шутка', 'юмор']
        scores['punchlines_wit'] = sum(1 for keyword in wit_keywords if keyword in text) * 1.6

        # Цитатность/мемность
        meme_keywords = ['запоминающаяся', 'афоризм', 'каламбур', 'слоган', 'крылатая фраза']
        scores['quotability_memes'] = sum(1 for keyword in meme_keywords if keyword in text) * 1.4

        # Ставки и цель
        stakes_keywords = ['если', 'то', 'последний шанс', 'мы либо', 'либо', 'ставки', 'цель']
        scores['stakes_goals'] = sum(1 for keyword in stakes_keywords if keyword in text) * 1.2

        # Крючки/клиффхэнгеры
        hook_keywords = ['вопрос', 'недосказанность', 'развязка', 'продолжение', 'что дальше']
        scores['hooks_cliffhangers'] = sum(1 for keyword in hook_keywords if keyword in text) * 1.0

        # Штраф за визуальную зависимость (отрицательный)
        visual_keywords = ['визуально', 'зрительно', 'видно', 'картинка', 'изображение']
        scores['visual_penalty'] = -sum(1 for keyword in visual_keywords if keyword in text) * 0.5

        # Нормализация оценок к шкале 0-10
        for key in scores:
            scores[key] = min(max(scores[key], 0), 10)

        return scores

    def _trim_boring_segments(self, ranked_moments: List[RankedMoment], transcription_data: Dict[str, Any]) -> List[RankedMoment]:
        """Обрезка скучных секунд"""
        try:
            trimmed_moments = []

            for rm in ranked_moments:
                moment = rm.moment

                # Поиск скучных сегментов в моменте
                boring_segments = self._detect_boring_segments_in_moment(moment, transcription_data)

                if boring_segments:
                    # Обрезка скучных сегментов
                    trimmed_moment = self._apply_trimming(moment, boring_segments)
                    if trimmed_moment:
                        rm.moment = trimmed_moment
                        logger.logger.debug(f"Обрезан момент {rm.rank}: {len(boring_segments)} скучных сегментов")

                trimmed_moments.append(rm)

            return trimmed_moments

        except Exception as e:
            logger.logger.error(f"Ошибка при обрезке скучных сегментов: {e}")
            return ranked_moments

    def _detect_boring_segments_in_moment(self, moment: FilmMoment, transcription_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Обнаружение скучных сегментов в моменте"""
        boring_segments = []
        threshold = self.film_config.pause_threshold

        segments = transcription_data.get('segments', [])

        for seg in segments:
            if not isinstance(seg, (list, tuple)) or len(seg) < 3:
                continue

            seg_start = float(seg[1])
            seg_end = float(seg[2])
            seg_text = str(seg[0]).strip()

            # Проверка на вхождение в момент
            if not (seg_start >= moment.start_time and seg_end <= moment.end_time):
                continue

            # Критерии скучного сегмента
            duration = seg_end - seg_start

            # Длинные паузы
            if duration > threshold:
                boring_segments.append({
                    'start': seg_start,
                    'end': seg_end,
                    'reason': 'long_pause',
                    'duration': duration
                })
                continue

            # Филлеры
            filler_words = ['э-э', 'м-м', 'ну', 'эээ', 'гм', 'кхм']
            if any(filler in seg_text.lower() for filler in filler_words):
                boring_segments.append({
                    'start': seg_start,
                    'end': seg_end,
                    'reason': 'filler_words',
                    'text': seg_text
                })

        return boring_segments

    def _apply_trimming(self, moment: FilmMoment, boring_segments: List[Dict[str, Any]]) -> Optional[FilmMoment]:
        """Применение обрезки к моменту"""
        if not boring_segments:
            return moment

        # Сортировка скучных сегментов по времени
        boring_segments.sort(key=lambda x: x['start'])

        # Создание новых границ момента, исключая скучные сегменты
        new_segments = []
        current_start = moment.start_time

        for boring in boring_segments:
            # Добавляем сегмент до скучного
            if current_start < boring['start']:
                new_segments.append({
                    'start': current_start,
                    'end': boring['start'],
                    'text': 'content'
                })

            # Переходим к следующему после скучного
            current_start = boring['end']

        # Добавляем оставшийся сегмент
        if current_start < moment.end_time:
            new_segments.append({
                'start': current_start,
                'end': moment.end_time,
                'text': 'content'
            })

        # Фильтруем слишком короткие сегменты
        new_segments = [s for s in new_segments if s['end'] - s['start'] > 1.0]

        if not new_segments:
            return None

        # Обновляем момент
        total_duration = sum(s['end'] - s['start'] for s in new_segments)

        # Проверяем минимальную длительность
        min_duration = self.film_config.combo_duration[0] if moment.moment_type == 'COMBO' else self.film_config.single_duration[0]
        if total_duration < min_duration:
            return None

        # Создаем новый момент с объединенными сегментами
        return FilmMoment(
            moment_type=moment.moment_type,
            start_time=new_segments[0]['start'],
            end_time=new_segments[-1]['end'],
            text=moment.text,
            segments=moment.segments,
            context=f"{moment.context} (обрезан: {len(boring_segments)} скучных сегментов)"
        )

    def _create_result(self, video_path: str, ranked_moments: List[RankedMoment], transcription_data: Dict[str, Any]) -> FilmAnalysisResult:
        """Создание финального результата анализа"""
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        duration = transcription_data.get('duration', 0)

        # Формирование keep_ranges
        keep_ranges = []
        for rm in ranked_moments[:10]:  # Топ-10 моментов
            keep_ranges.append({
                'start': rm.moment.start_time,
                'end': rm.moment.end_time,
                'type': rm.moment.moment_type,
                'score': round(rm.total_score, 2),
                'text': rm.moment.text[:200] + '...' if len(rm.moment.text) > 200 else rm.moment.text
            })

        # Формирование scores
        scores = []
        for rm in ranked_moments[:10]:
            score_dict = {
                'moment_id': f"{rm.moment.moment_type.lower()}_{rm.rank}",
                'total': round(rm.total_score, 2)
            }
            score_dict.update({k: round(v, 2) for k, v in rm.scores.items()})
            scores.append(score_dict)

        # Превью текста
        total_duration = sum(r['end'] - r['start'] for r in keep_ranges)
        preview_text = f"Фильм содержит {len(keep_ranges)} лучших моментов длительностью {total_duration:.1f} минут из оригинальных {duration/60:.1f} минут видео."

        # Риски
        risks = []
        if len(keep_ranges) < 3:
            risks.append("Мало подходящих моментов найдено")
        if duration > 3600:  # > 1 час
            risks.append("Длинное видео - возможны проблемы с обработкой")
        if not keep_ranges:
            risks.append("Не найдено ни одного подходящего момента")

        # Метаданные
        metadata = {
            'processed_at': datetime.now().isoformat(),
            'model_version': self.config.llm.model_name,
            'total_segments_analyzed': len(transcription_data.get('segments', [])),
            'video_duration': duration,
            'moments_found': len(ranked_moments)
        }

        return FilmAnalysisResult(
            video_id=video_id,
            duration=duration,
            keep_ranges=keep_ranges,
            scores=scores,
            preview_text=preview_text,
            risks=risks,
            metadata=metadata
        )

    def _create_empty_result(self, video_path: str) -> FilmAnalysisResult:
        """Создание пустого результата при ошибке"""
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        return FilmAnalysisResult(
            video_id=video_id,
            duration=0,
            keep_ranges=[],
            scores=[],
            preview_text="Не удалось проанализировать видео",
            risks=["Ошибка обработки видео"],
            metadata={'error': True}
        )


def scan_movies_folder() -> List[str]:
    """
    Сканирует папку movies и возвращает список видео файлов.
    Поддерживаемые форматы: .mp4, .avi, .mkv, .mov, .wmv

    Returns:
        List[str]: Список путей к видео файлам
    """
    movies_dir = os.path.join(os.getcwd(), "movies")
    supported_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv'}

    if not os.path.exists(movies_dir):
        logger.logger.warning(f"Папка movies не найдена: {movies_dir}")
        return []

    video_files = []
    try:
        for file in os.listdir(movies_dir):
            if os.path.isfile(os.path.join(movies_dir, file)):
                _, ext = os.path.splitext(file.lower())
                if ext in supported_extensions:
                    video_files.append(os.path.join(movies_dir, file))
    except Exception as e:
        logger.logger.error(f"Ошибка при сканировании папки movies: {e}")
        return []

    return sorted(video_files)


def display_movie_selection(video_files: List[str]) -> None:
    """
    Отображает список видео файлов с номерами для выбора.

    Args:
        video_files: Список путей к видео файлам
    """
    if not video_files:
        print("\n📁 Папка movies пуста или не содержит поддерживаемых видео файлов.")
        print("Поддерживаемые форматы: .mp4, .avi, .mkv, .mov, .wmv")
        print("Поместите видео файлы в папку 'movies' в корне проекта.")
        return

    print(f"\n🎬 Найдено {len(video_files)} видео файлов в папке movies:")
    print("-" * 60)

    for i, file_path in enumerate(video_files, 1):
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

        # Получаем информацию о длительности, если возможно
        duration_str = ""
        try:
            from Components.Edit import get_video_dimensions
            # Для получения длительности можно использовать ffprobe
            duration = _get_video_duration_quick(file_path)
            if duration and duration > 0:
                duration_str = f" ({duration:.1f} мин)"
        except:
            pass

        print("2d")

    print("-" * 60)
    print("0. Вернуться в главное меню")
    print("URL. Ввести YouTube URL или путь к файлу вручную")


def select_movie_by_number(video_files: List[str]) -> Optional[str]:
    """
    Позволяет пользователю выбрать видео файл по номеру.

    Args:
        video_files: Список путей к видео файлам

    Returns:
        Optional[str]: Выбранный путь к файлу или None при отмене
    """
    if not video_files:
        return None

    while True:
        try:
            choice = input("\nВведите номер видео (1-{}) или 0 для отмены: ".format(len(video_files))).strip()

            if choice == "0":
                return None

            if choice.upper() == "URL":
                # Возвращаем специальный маркер для ручного ввода
                return "URL_INPUT"

            choice_num = int(choice)

            if 1 <= choice_num <= len(video_files):
                selected_file = video_files[choice_num - 1]

                # Проверяем, что файл все еще существует
                if not os.path.exists(selected_file):
                    print(f"❌ Файл не найден: {os.path.basename(selected_file)}")
                    print("Файл мог быть удален или перемещен.")
                    return None

                file_name = os.path.basename(selected_file)
                file_size = os.path.getsize(selected_file) / (1024 * 1024)  # MB
                print(f"\n✅ Выбрано: {file_name} ({file_size:.1f} MB)")
                return selected_file
            else:
                print(f"❌ Неверный номер. Введите число от 1 до {len(video_files)}")

        except ValueError:
            print("❌ Неверный ввод. Введите число или 'URL' для ручного ввода")
        except KeyboardInterrupt:
            print("\n\nОтмена выбора.")
            return None
        except Exception as e:
            print(f"❌ Ошибка при выборе файла: {e}")
            return None


def _get_video_duration_quick(video_path: str) -> Optional[float]:
    """
    Быстрое получение длительности видео через ffprobe.

    Args:
        video_path: Путь к видео файлу

    Returns:
        Optional[float]: Длительность в минутах или None при ошибке
    """
    try:
        import subprocess
        import json

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return duration / 60  # в минутах
        else:
            return None
    except Exception:
        return None


def analyze_film_main(url: Optional[str] = None, local_path: Optional[str] = None) -> FilmAnalysisResult:
    """
    Основная функция для анализа фильма.
    Точка входа для интеграции с main.py
    """
    try:
        config = get_config()
        analyzer = FilmAnalyzer(config)
        return analyzer.analyze_film(url, local_path)
    except Exception as e:
        logger.logger.error(f"Ошибка в analyze_film_main: {e}")
        # Возвращаем пустой результат
        return FilmAnalysisResult(
            video_id="error",
            duration=0,
            keep_ranges=[],
            scores=[],
            preview_text=f"Ошибка анализа: {str(e)}",
            risks=[str(e)],
            metadata={'error': True}
        )