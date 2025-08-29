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
from Components.Edit import crop_video, burn_captions, crop_bottom_video, animate_captions, get_video_dimensions
from Components.FaceCrop import crop_to_70_percent_with_blur, crop_to_vertical_average_face
from Components.Paths import build_short_output_name
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
    generated_shorts: List[str] = field(default_factory=list)  # Пути к сгенерированным шортам


class FilmAnalyzer:
    """
    Анализатор фильмов для выделения лучших моментов.
    Интегрируется с существующими компонентами проекта.
    """

    def __init__(self, config: AppConfig):
        logger.logger.info("Инициализация FilmAnalyzer...")
        self.config = config
        logger.logger.info("Создание подключения к базе данных VideoDatabase...")
        self.db = VideoDatabase()
        logger.logger.info("✅ FilmAnalyzer инициализирован успешно")

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
        6. Генерация шортов (если включено)
        7. Формирование результата
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
        if not ranked_moments:
            logger.logger.warning("После фильтрации по качеству не осталось подходящих моментов")
            return self._create_empty_result(video_path)

        # 4. Обрезка скучных секунд
        trimmed_moments = self._trim_boring_segments(ranked_moments, transcription_data)

        # 5. Генерация шортов (если включено)
        generated_shorts = []
        logger.logger.info(f"Проверка условий для генерации шортов:")
        logger.logger.info(f"  generate_shorts: {self.film_config.generate_shorts}")
        logger.logger.info(f"  количество моментов: {len(trimmed_moments)}")

        if self.film_config.generate_shorts and trimmed_moments:
            logger.logger.info("✅ Условия выполнены, начинаем генерацию шортов...")
            generated_shorts = self._generate_shorts_from_moments(video_path, trimmed_moments, transcription_data)
            logger.logger.info(f"Сгенерировано {len(generated_shorts)} шортов")
        else:
            if not self.film_config.generate_shorts:
                logger.logger.warning("⚠️ Генерация шортов отключена в конфигурации")
            if not trimmed_moments:
                logger.logger.warning("⚠️ Нет подходящих моментов для генерации шортов")

        # 6. Формирование результата
        result = self._create_result(video_path, trimmed_moments, transcription_data, generated_shorts)

        logger.logger.info(f"Анализ фильма завершен. Найдено {len(trimmed_moments)} моментов, сгенерировано {len(generated_shorts)} шортов")
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
            logger.logger.info("Формирование текста транскрибации через build_transcription_prompt...")
            transcription_text = build_transcription_prompt(segments_dict)
            logger.logger.info(f"✅ Текст транскрибации сформирован: {len(transcription_text)} символов")

            # Анализ через LLM
            logger.logger.info("Анализ моментов через LLM...")
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

            logger.logger.info("Создание конфигурации генерации через make_generation_config...")
            generation_config = make_generation_config(system_instruction, temperature=0.3)
            logger.logger.info("✅ Конфигурация генерации создана")

            logger.logger.info("Отправка запроса к LLM через call_llm_with_retry...")
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

            # Фильтрация по минимальному порогу качества
            if total_score < self.film_config.min_quality_score:
                logger.logger.debug(f"Момент {i+1} отфильтрован по порогу качества: {total_score:.2f} < {self.film_config.min_quality_score}")
                continue

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

        logger.logger.info(f"Ранжирование завершено. Найдено {len(ranked_moments)} моментов с качеством >= {self.film_config.min_quality_score}")
        if ranked_moments:
            logger.logger.info(f"Лучший момент имеет балл {ranked_moments[0].total_score:.2f}")

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

            # Проверка на ПЕРЕСЕЧЕНИЕ с моментом (не полное вхождение!)
            # Сегмент должен пересекаться с моментом хотя бы частично
            if not (seg_end > moment.start_time and seg_start < moment.end_time):
                continue

            # Ограничиваем сегмент границами момента для корректного анализа
            effective_start = max(seg_start, moment.start_time)
            effective_end = min(seg_end, moment.end_time)
            duration = effective_end - effective_start

            # Критерии скучного сегмента
            # Длинные паузы
            if duration > threshold:
                boring_segments.append({
                    'start': effective_start,
                    'end': effective_end,
                    'reason': 'long_pause',
                    'duration': duration
                })
                continue

            # Филлеры
            filler_words = self.film_config.filler_words
            if any(filler in seg_text.lower() for filler in filler_words):
                boring_segments.append({
                    'start': effective_start,
                    'end': effective_end,
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

    def _generate_shorts_from_moments(self, video_path: str, ranked_moments: List[RankedMoment], transcription_data: Dict[str, Any]) -> List[str]:
        """Генерация шортов из найденных моментов"""
        logger.logger.info("=== НАЧАЛО ГЕНЕРАЦИИ ШОРТОВ ===")

        # 1. Логирование входных параметров
        logger.logger.info("--- ВХОДНЫЕ ПАРАМЕТРЫ ---")
        logger.logger.info(f"video_path: {video_path}")
        logger.logger.info(f"video_path существует: {os.path.exists(video_path)}")
        if os.path.exists(video_path):
            video_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            logger.logger.info(f"Размер видео файла: {video_size:.2f} MB")

        logger.logger.info(f"ranked_moments: {len(ranked_moments)} моментов")
        for i, rm in enumerate(ranked_moments):
            moment = rm.moment
            logger.logger.debug(f"  Момент {i+1}: {moment.moment_type} {moment.start_time:.2f}s-{moment.end_time:.2f}s, score={rm.total_score:.2f}")

        logger.logger.info(f"transcription_data ключи: {list(transcription_data.keys())}")
        if 'segments' in transcription_data:
            logger.logger.info(f"Количество сегментов транскрибации: {len(transcription_data['segments'])}")
        if 'duration' in transcription_data:
            logger.logger.info(f"Длительность видео по транскрибации: {transcription_data['duration']:.2f}s")

        # 2. Валидация входных данных
        logger.logger.info("--- ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ ---")
        if not video_path or not os.path.exists(video_path):
            logger.logger.error(f"❌ Видео файл не найден: {video_path}")
            return []

        if not ranked_moments:
            logger.logger.warning("⚠️ Нет моментов для обработки")
            return []

        if not transcription_data:
            logger.logger.warning("⚠️ Нет данных транскрибации")
            return []

        logger.logger.info("✅ Валидация входных данных пройдена")

        generated_shorts = []
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        logger.logger.info(f"Извлечен video_id: {video_id}")

        # 3. Получение размеров видео для валидации
        logger.logger.info("--- ПОЛУЧЕНИЕ РАЗМЕРОВ ВИДЕО ---")
        try:
            logger.logger.debug(f"Вызов get_video_dimensions для файла: {video_path}")
            initial_width, initial_height = get_video_dimensions(video_path)
            logger.logger.info(f"✅ Размеры видео получены: {initial_width}x{initial_height}")
            if not initial_width or not initial_height:
                logger.logger.error("❌ Не удалось получить корректные размеры видео (один из размеров равен 0 или None)")
                return []
        except Exception as e:
            logger.logger.error(f"❌ Ошибка при получении размеров видео: {e}")
            import traceback
            logger.logger.error(f"Traceback: {traceback.format_exc()}")
            return []

        # 4. Выбор топ моментов для обработки
        logger.logger.info("--- ВЫБОР ТОП МОМЕНТОВ ---")
        max_moments = 10
        top_moments = ranked_moments[:max_moments]
        logger.logger.info(f"Выбрано топ-{len(top_moments)} моментов из {len(ranked_moments)} (максимум {max_moments})")

        # 5. Дополнительная валидация моментов
        logger.logger.info("--- ДОПОЛНИТЕЛЬНАЯ ВАЛИДАЦИЯ МОМЕНТОВ ---")

        # Извлекаем только FilmMoment объекты для валидации
        moments_to_validate = [rm.moment for rm in top_moments]
        video_duration = transcription_data.get('duration', None)

        # Вызываем метод валидации
        valid_moments_data = self._validate_moment_data(moments_to_validate, video_duration)

        # Создаем список RankedMoment только из валидных моментов
        valid_moments = []
        for rm in top_moments:
            if rm.moment in valid_moments_data:
                valid_moments.append(rm)

        if not valid_moments:
            logger.logger.error("❌ Нет валидных моментов для генерации шортов после всех проверок")
            return []

        top_moments = valid_moments
        logger.logger.info(f"✅ После валидации: {len(top_moments)} моментов готовы к обработке")

        # 6. Создание контекста обработки
        logger.logger.info("--- СОЗДАНИЕ КОНТЕКСТА ОБРАБОТКИ ---")
        processing_context = self._create_processing_context_for_moment(
            video_path, video_id, transcription_data, initial_width, initial_height
        )
        logger.logger.info("✅ Контекст обработки создан")

        # 7. Основной цикл обработки моментов
        logger.logger.info("--- НАЧАЛО ОБРАБОТКИ МОМЕНТОВ ---")
        for i, rm in enumerate(top_moments):
            try:
                moment = rm.moment
                logger.logger.info(f"=== ОБРАБОТКА МОМЕНТА {i+1}/{len(top_moments)} ===")
                logger.logger.info(f"Момент {rm.rank}: таймкод {moment.start_time:.2f}s - {moment.end_time:.2f}s")
                logger.logger.info(f"Длительность момента: {moment.end_time - moment.start_time:.2f}s")
                logger.logger.info(f"Тип момента: {moment.moment_type}")
                logger.logger.info(f"Балл: {rm.total_score:.2f}")
                logger.logger.info(f"Текст: {moment.text[:200]}...")
                if len(moment.text) > 200:
                    logger.logger.debug(f"Полный текст: {moment.text}")

                # Финальная проверка корректности таймкодов
                if moment.start_time >= moment.end_time:
                    logger.logger.error(f"❌ Некорректные таймкоды: start={moment.start_time} >= end={moment.end_time}")
                    continue

                duration = moment.end_time - moment.start_time
                if duration < 1.0:
                    logger.logger.warning(f"⚠️ Момент слишком короткий: {duration:.2f}s (минимум 1.0s)")

                # Создаем структуру highlight для совместимости с process_highlight
                highlight_item = {
                    'start': moment.start_time,
                    'end': moment.end_time,
                    'caption_with_hashtags': f"Film Moment {rm.rank}: {moment.text[:100]}...",
                    'segment_text': moment.text,
                    '_seq': i + 1,
                    '_total': len(top_moments)
                }
                logger.logger.debug(f"Создан highlight_item: {highlight_item}")

                # Генерируем шорт
                logger.logger.info(f"Вызов _process_moment_to_short для момента {rm.rank}")
                short_path = self._process_moment_to_short(processing_context, highlight_item, i + 1)

                if short_path:
                    generated_shorts.append(short_path)
                    logger.logger.info(f"✅ УСПЕШНО сгенерирован шорт: {short_path}")

                    # Проверяем, что файл действительно создан
                    if os.path.exists(short_path):
                        file_size = os.path.getsize(short_path) / (1024 * 1024)  # MB
                        logger.logger.info(f"✅ Файл шорта создан: {file_size:.2f} MB")
                    else:
                        logger.logger.warning(f"⚠️ Файл шорта не найден после создания: {short_path}")
                else:
                    logger.logger.error(f"❌ НЕ УДАЛОСЬ сгенерировать шорт для момента {rm.rank}")

            except Exception as e:
                logger.logger.error(f"❌ ОШИБКА при генерации шорта для момента {rm.rank}: {e}")
                import traceback
                logger.logger.error(f"Traceback: {traceback.format_exc()}")
                continue

        # 8. Завершение генерации шортов
        logger.logger.info("=== ЗАВЕРШЕНИЕ ГЕНЕРАЦИИ ШОРТОВ ===")
        logger.logger.info(f"Всего сгенерировано шортов: {len(generated_shorts)}")

        if generated_shorts:
            logger.logger.info("Список сгенерированных шортов:")
            for i, short_path in enumerate(generated_shorts, 1):
                exists = os.path.exists(short_path)
                size = os.path.getsize(short_path) / (1024 * 1024) if exists else 0
                logger.logger.info(f"  {i}. {short_path} ({size:.2f} MB, существует: {exists})")
        else:
            logger.logger.warning("⚠️ Ни один шорт не был сгенерирован")

        logger.logger.info("✅ Генерация шортов из моментов завершена")
        return generated_shorts

    def _validate_moment_data(self, moments: List[FilmMoment], video_duration: float = None) -> List[FilmMoment]:
        """Валидация данных моментов с подробным логированием"""
        logger.logger.info("=== ВАЛИДАЦИЯ ДАННЫХ МОМЕНТОВ ===")
        logger.logger.info(f"Вход: {len(moments)} моментов, длительность видео: {video_duration}")

        if not moments:
            logger.logger.warning("⚠️ Список моментов пуст")
            return []

        valid_moments = []
        invalid_count = 0

        for i, moment in enumerate(moments):
            logger.logger.debug(f"Валидация момента {i+1}: {moment.moment_type} {moment.start_time:.2f}s-{moment.end_time:.2f}s")

            is_valid = True
            issues = []

            # Проверка типа момента
            if moment.moment_type not in ['COMBO', 'SINGLE']:
                issues.append(f"неверный тип момента: {moment.moment_type}")
                is_valid = False

            # Проверка таймкодов
            if moment.start_time < 0:
                issues.append(f"отрицательное время начала: {moment.start_time}")
                is_valid = False

            if moment.end_time <= moment.start_time:
                issues.append(f"некорректный интервал: {moment.start_time} >= {moment.end_time}")
                is_valid = False

            # Проверка длительности
            duration = moment.end_time - moment.start_time
            if duration < 1.0:
                issues.append(f"слишком короткий: {duration:.2f}s (минимум 1.0s)")
                is_valid = False
            elif duration > 120.0:  # максимум 2 минуты
                issues.append(f"слишком длинный: {duration:.2f}s (максимум 120.0s)")
                is_valid = False

            # Проверка границ видео
            if video_duration and moment.end_time > video_duration:
                issues.append(f"выходит за границы видео: {moment.end_time:.2f}s > {video_duration:.2f}s")
                # Корректируем вместо инвалидации
                moment.end_time = video_duration
                logger.logger.info(f"  Скорректирован конец момента до {moment.end_time:.2f}s")

            # Проверка текста
            if not moment.text or not moment.text.strip():
                issues.append("пустой текст")
                is_valid = False

            # Проверка сегментов для COMBO
            if moment.moment_type == 'COMBO':
                if not moment.segments:
                    issues.append("COMBO момент без суб-сегментов")
                    is_valid = False
                elif len(moment.segments) < 2:
                    issues.append(f"COMBO момент с недостаточным количеством сегментов: {len(moment.segments)}")
                    is_valid = False

            if is_valid:
                valid_moments.append(moment)
                logger.logger.debug(f"✅ Момент {i+1} прошел валидацию")
            else:
                invalid_count += 1
                logger.logger.warning(f"❌ Момент {i+1} отклонен: {', '.join(issues)}")

        logger.logger.info(f"Результат валидации: {len(valid_moments)} валидных, {invalid_count} отклоненных из {len(moments)}")
        return valid_moments

    def _extract_video_segment(self, input_path: str, output_path: str, start_time: float, end_time: float, width: int, height: int) -> bool:
        """Извлечение сегмента видео с подробным логированием"""
        logger.logger.info("=== ИЗВЛЕЧЕНИЕ СЕГМЕНТА ВИДЕО ===")
        logger.logger.info(f"Входной файл: {input_path}")
        logger.logger.info(f"Выходной файл: {output_path}")
        logger.logger.info(f"Таймкоды: {start_time:.2f}s - {end_time:.2f}s")
        logger.logger.info(f"Размеры: {width}x{height}")

        # 1. Валидация входных параметров
        logger.logger.debug("Валидация входных параметров...")
        if not input_path or not os.path.exists(input_path):
            logger.logger.error(f"❌ Входной файл не найден: {input_path}")
            return False

        if not output_path:
            logger.logger.error("❌ Не указан выходной файл")
            return False

        if start_time < 0:
            logger.logger.error(f"❌ Отрицательное время начала: {start_time}")
            return False

        if end_time <= start_time:
            logger.logger.error(f"❌ Некорректный интервал: {start_time} >= {end_time}")
            return False

        duration = end_time - start_time
        if duration < 0.1:
            logger.logger.error(f"❌ Слишком короткий сегмент: {duration:.2f}s")
            return False

        logger.logger.info(f"✅ Валидация параметров пройдена, длительность сегмента: {duration:.2f}s")

        # 2. Проверка входного файла
        try:
            input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
            logger.logger.info(f"Размер входного файла: {input_size:.2f} MB")
        except Exception as e:
            logger.logger.warning(f"Не удалось получить размер входного файла: {e}")

        # 3. Вызов crop_video
        logger.logger.info("Вызов crop_video для извлечения сегмента...")
        try:
            success = crop_video(input_path, output_path, start_time, end_time, width, height)

            if success:
                logger.logger.info("✅ crop_video выполнен успешно")
            else:
                logger.logger.error("❌ crop_video вернул False")
                return False

        except Exception as e:
            logger.logger.error(f"❌ Исключение при вызове crop_video: {e}")
            import traceback
            logger.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

        # 4. Проверка результата
        logger.logger.info("Проверка созданного файла...")
        if not os.path.exists(output_path):
            logger.logger.error(f"❌ Выходной файл не найден: {output_path}")
            return False

        try:
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            logger.logger.info(f"✅ Выходной файл создан: {output_path} ({output_size:.2f} MB)")

            # Проверка на пустой файл
            if output_size < 0.1:  # менее 100KB
                logger.logger.error(f"❌ Выходной файл слишком маленький: {output_size:.2f} MB")
                try:
                    os.remove(output_path)
                    logger.logger.info(f"Удален пустой файл: {output_path}")
                except Exception as e:
                    logger.logger.warning(f"Не удалось удалить пустой файл: {e}")
                return False

            logger.logger.info("✅ Сегмент видео успешно извлечен и проверен")
            return True

        except Exception as e:
            logger.logger.error(f"❌ Ошибка при проверке выходного файла: {e}")
            return False

    def _add_captions_to_short(self, input_video: str, output_video: str, transcription_segments: List[List], start_time: float, end_time: float, style_cfg=None) -> bool:
        """Добавление субтитров к видео с подробным логированием"""
        logger.logger.info("=== ДОБАВЛЕНИЕ СУБТИТРОВ К ВИДЕО ===")
        logger.logger.info(f"Входное видео: {input_video}")
        logger.logger.info(f"Выходное видео: {output_video}")
        logger.logger.info(f"Таймкоды: {start_time:.2f}s - {end_time:.2f}s")
        logger.logger.info(f"Количество сегментов транскрибации: {len(transcription_segments)}")

        # 1. Валидация входных параметров
        logger.logger.debug("Валидация входных параметров...")
        if not input_video or not os.path.exists(input_video):
            logger.logger.error(f"❌ Входное видео не найдено: {input_video}")
            return False

        if not output_video:
            logger.logger.error("❌ Не указан путь выходного видео")
            return False

        if not transcription_segments:
            logger.logger.warning("⚠️ Нет сегментов транскрибации для субтитров")
            # Всё равно пытаемся создать видео без субтитров
            logger.logger.info("Попытка создать видео без субтитров...")

        logger.logger.info(f"✅ Валидация параметров пройдена")

        # 2. Фильтрация релевантных сегментов
        if transcription_segments:
            logger.logger.info("Фильтрация релевантных сегментов...")
            relevant_segments = []
            for seg in transcription_segments:
                if len(seg) >= 3:
                    seg_start = float(seg[1])
                    seg_end = float(seg[2])
                    # Проверяем пересечение с моментом
                    if seg_end > start_time and seg_start < end_time:
                        relevant_segments.append(seg)

            logger.logger.info(f"Релевантных сегментов: {len(relevant_segments)} из {len(transcription_segments)}")
            if relevant_segments:
                logger.logger.debug(f"Примеры релевантных сегментов: {relevant_segments[:2]}")

            transcription_segments = relevant_segments
        else:
            logger.logger.warning("⚠️ Сегменты транскрибации отсутствуют")

        # 3. Вызов burn_captions
        logger.logger.info("Вызов burn_captions...")
        try:
            # Создаем временный файл для аудио (как в оригинальном коде)
            temp_audio = input_video  # Используем входное видео как источник аудио

            success = burn_captions(
                input_video, temp_audio, transcription_segments,
                start_time, end_time, output_video, style_cfg=style_cfg
            )

            if success:
                logger.logger.info("✅ burn_captions выполнен успешно")
            else:
                logger.logger.error("❌ burn_captions вернул False")
                return False

        except Exception as e:
            logger.logger.error(f"❌ Исключение при вызове burn_captions: {e}")
            import traceback
            logger.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

        # 4. Проверка результата
        logger.logger.info("Проверка созданного видео с субтитрами...")
        if not os.path.exists(output_video):
            logger.logger.error(f"❌ Выходное видео не найдено: {output_video}")
            return False

        try:
            output_size = os.path.getsize(output_video) / (1024 * 1024)  # MB
            logger.logger.info(f"✅ Видео с субтитрами создано: {output_video} ({output_size:.2f} MB)")

            # Проверка на пустой файл
            if output_size < 0.1:  # менее 100KB
                logger.logger.error(f"❌ Выходное видео слишком маленькое: {output_size:.2f} MB")
                try:
                    os.remove(output_video)
                    logger.logger.info(f"Удален пустой файл: {output_video}")
                except Exception as e:
                    logger.logger.warning(f"Не удалось удалить пустой файл: {e}")
                return False

            logger.logger.info("✅ Субтитры успешно добавлены к видео")
            return True

        except Exception as e:
            logger.logger.error(f"❌ Ошибка при проверке выходного видео: {e}")
            return False

    def _create_processing_context_for_moment(self, video_path: str, video_id: str, transcription_data: Dict[str, Any], width: int, height: int):
        """Создание контекста обработки для генерации шорта"""
        # Импортируем необходимые классы
        from Components.Database import VideoDatabase

        class MockProcessingContext:
            def __init__(self, video_path, video_id, transcription_data, width, height, config):
                self.video_path = video_path
                self.video_id = video_id
                self.transcription_segments = transcription_data.get('segments', [])
                self.word_level_transcription = transcription_data.get('word_level', {})
                self.initial_width = width
                self.initial_height = height
                self.cfg = config
                self.db = VideoDatabase()
                self.outputs = []

        return MockProcessingContext(video_path, video_id, transcription_data, width, height, self.config)

    def _process_moment_to_short(self, ctx, highlight_item, seq: int) -> Optional[str]:
        """Обработка момента в шорт (упрощенная версия process_highlight)"""
        logger.logger.info(f"--- НАЧАЛО ОБРАБОТКИ МОМЕНТА {seq} ---")

        try:
            # 1. Извлечение и валидация таймкодов
            logger.logger.debug("--- ИЗВЛЕЧЕНИЕ ТАЙМКОДОВ ---")
            start = float(highlight_item["start"])
            stop = float(highlight_item["end"])
            logger.logger.info(f"Исходные таймкоды: {start:.2f}s - {stop:.2f}s")

            # Корректировка длительности
            adjusted_stop = stop
            if adjusted_stop <= start:
                adjusted_stop = start + 1.0
                logger.logger.warning(f"⚠️ Скорректирована длительность: {start:.2f}s - {adjusted_stop:.2f}s (было <= start)")

            duration = adjusted_stop - start
            logger.logger.info(f"Финальные таймкоды: {start:.2f}s - {adjusted_stop:.2f}s")
            logger.logger.info(f"Длительность сегмента: {duration:.2f}s")

            # Проверяем границы видео
            if ctx.word_level_transcription and 'segments' in ctx.word_level_transcription:
                video_duration = len(ctx.word_level_transcription['segments']) * 0.1  # приблизительно
                if adjusted_stop > video_duration:
                    logger.logger.warning(f"⚠️ Конец сегмента {adjusted_stop:.2f}s выходит за длительность видео {video_duration:.2f}s")

            # 2. Определение путей файлов
            logger.logger.info("--- ОПРЕДЕЛЕНИЕ ПУТЕЙ ФАЙЛОВ ---")
            base_name = os.path.splitext(os.path.basename(ctx.video_path))[0]
            output_base = f"{base_name}_film_moment_{seq}"
            logger.logger.debug(f"base_name: {base_name}, output_base: {output_base}")

            temp_segment = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_temp.mp4")
            cropped_vertical = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_vertical.mp4")
            final_output, _ = build_short_output_name(base_name, seq, ctx.cfg.processing.shorts_dir, prefix="film_moment")
            logger.logger.info(f"✅ Сформирован путь финального файла через build_short_output_name: {final_output}")

            logger.logger.info("Пути файлов:")
            logger.logger.info(f"  Исходное видео: {ctx.video_path}")
            logger.logger.info(f"  Временный сегмент: {temp_segment}")
            logger.logger.info(f"  Вертикальный кроп: {cropped_vertical}")
            logger.logger.info(f"  Финальный шорт: {final_output}")

            # 3. Проверка и создание директорий
            logger.logger.info("--- ПРОВЕРКА ДИРЕКТОРИЙ ---")
            videos_dir = ctx.cfg.processing.videos_dir
            shorts_dir = ctx.cfg.processing.shorts_dir
            logger.logger.info(f"Директории: videos_dir={videos_dir}, shorts_dir={shorts_dir}")

            # Проверяем videos_dir
            if not os.path.exists(videos_dir):
                logger.logger.warning(f"⚠️ Директория videos_dir не существует: {videos_dir}")
                try:
                    os.makedirs(videos_dir, exist_ok=True)
                    logger.logger.info(f"✅ Создана директория videos_dir: {videos_dir}")
                except Exception as e:
                    logger.logger.error(f"❌ Не удалось создать videos_dir: {e}")
                    return None
            else:
                logger.logger.debug(f"✅ Директория videos_dir существует: {videos_dir}")

            # Проверяем shorts_dir
            if not os.path.exists(shorts_dir):
                logger.logger.warning(f"⚠️ Директория shorts_dir не существует: {shorts_dir}")
                try:
                    os.makedirs(shorts_dir, exist_ok=True)
                    logger.logger.info(f"✅ Создана директория shorts_dir: {shorts_dir}")
                except Exception as e:
                    logger.logger.error(f"❌ Не удалось создать shorts_dir: {e}")
                    return None
            else:
                logger.logger.debug(f"✅ Директория shorts_dir существует: {shorts_dir}")

            # 4. Извлечение сегмента видео
            extract_success = self._extract_video_segment(
                ctx.video_path, temp_segment, start, adjusted_stop, ctx.initial_width, ctx.initial_height
            )

            if not extract_success:
                logger.logger.error(f"❌ Не удалось извлечь сегмент для момента {seq}")
                return None

            logger.logger.info("✅ Шаг 1 УСПЕШЕН: Сегмент извлечен и проверен")

            # 5. Создание вертикального кропа
            logger.logger.info("--- ШАГ 2: СОЗДАНИЕ ВЕРТИКАЛЬНОГО КРОПА ---")
            crop_mode = ctx.cfg.processing.crop_mode
            logger.logger.info(f"Режим кропа: {crop_mode}")
            logger.logger.info(f"Входной файл: {temp_segment}")
            logger.logger.info(f"Выходной файл: {cropped_vertical}")

            # Проверяем существование входного файла для кропа
            if not os.path.exists(temp_segment):
                logger.logger.error(f"❌ Входной файл для кропа не найден: {temp_segment}")
                return None
            else:
                input_size = os.path.getsize(temp_segment) / (1024 * 1024)  # MB
                logger.logger.debug(f"✅ Входной файл существует: {input_size:.2f} MB")

            # Вызываем соответствующую функцию кропа
            try:
                if crop_mode == "70_percent_blur":
                    logger.logger.info("Вызов crop_to_70_percent_with_blur...")
                    crop_success = crop_to_70_percent_with_blur(temp_segment, cropped_vertical)
                    logger.logger.info(f"Результат crop_to_70_percent_with_blur: {crop_success}")
                elif crop_mode == "average_face":
                    logger.logger.info("Вызов crop_to_vertical_average_face...")
                    crop_success = crop_to_vertical_average_face(temp_segment, cropped_vertical)
                    logger.logger.info(f"Результат crop_to_vertical_average_face: {crop_success}")
                else:
                    logger.logger.warning(f"⚠️ Неизвестный режим кропа: {crop_mode}, используем 70_percent_blur")
                    crop_success = crop_to_70_percent_with_blur(temp_segment, cropped_vertical)
                    logger.logger.info(f"Результат crop_to_70_percent_with_blur (fallback): {crop_success}")

            except Exception as e:
                logger.logger.error(f"❌ Исключение при кропе: {e}")
                import traceback
                logger.logger.error(f"Traceback: {traceback.format_exc()}")
                crop_success = False

            if not crop_success:
                logger.logger.error(f"❌ Шаг 2 ПРОВАЛЕН: Не удалось создать вертикальный кроп для момента {seq}")
                # Очистка временных файлов
                if os.path.exists(temp_segment):
                    try:
                        os.remove(temp_segment)
                        logger.logger.info(f"Удален временный файл: {temp_segment}")
                    except Exception as e:
                        logger.logger.warning(f"Не удалось удалить временный файл: {e}")
                return None
            else:
                logger.logger.info("✅ Функция кропа выполнена успешно")

            # Проверяем результат кропа
            if not os.path.exists(cropped_vertical):
                logger.logger.error(f"❌ Файл вертикального кропа не найден: {cropped_vertical}")
                # Очистка
                if os.path.exists(temp_segment):
                    try:
                        os.remove(temp_segment)
                        logger.logger.info(f"Удален временный файл: {temp_segment}")
                    except Exception as e:
                        logger.logger.warning(f"Не удалось удалить временный файл: {e}")
                return None
            else:
                file_size = os.path.getsize(cropped_vertical) / (1024 * 1024)  # MB
                logger.logger.info(f"✅ Файл кропа создан: {file_size:.2f} MB")

                # Проверяем, что файл не пустой
                if file_size < 0.1:  # менее 100KB
                    logger.logger.error(f"❌ Файл кропа слишком маленький: {file_size:.2f} MB")
                    # Очистка
                    for temp_file in [temp_segment, cropped_vertical]:
                        if os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                                logger.logger.info(f"Удален пустой файл: {temp_file}")
                            except Exception as e:
                                logger.logger.warning(f"Не удалось удалить файл: {e}")
                    return None
                else:
                    logger.logger.info("✅ Шаг 2 УСПЕШЕН: Вертикальный кроп создан и проверен")

            # 6. Добавление субтитров
            logger.logger.info("--- ШАГ 3: ДОБАВЛЕНИЕ СУБТИТРОВ ---")

            # Проверяем существование входного файла для субтитров
            if not os.path.exists(cropped_vertical):
                logger.logger.error(f"❌ Входной файл для субтитров не найден: {cropped_vertical}")
                # Очистка
                for temp_file in [temp_segment]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                            logger.logger.info(f"Удален временный файл: {temp_file}")
                        except Exception as e:
                            logger.logger.warning(f"Не удалось удалить временный файл: {e}")
                return None

            # Подготовка данных транскрибации
            transcription_segments = [[
                str(seg.get("text", "")),
                float(seg.get("start", 0.0)),
                float(seg.get("end", 0.0)),
            ] for seg in (ctx.transcription_segments or [])]

            # Вызываем новый метод добавления субтитров
            caption_success = self._add_captions_to_short(
                cropped_vertical, final_output, transcription_segments,
                start, adjusted_stop, style_cfg=ctx.cfg.captions
            )

            # 7. Обработка результата субтитров и финализация
            logger.logger.info("--- ОБРАБОТКА РЕЗУЛЬТАТА СУБТИТРОВ ---")

            # Очистка временных файлов
            temp_files_to_clean = [temp_segment, cropped_vertical]
            logger.logger.info("Очистка временных файлов...")
            for temp_file in temp_files_to_clean:
                if os.path.exists(temp_file):
                    try:
                        file_size = os.path.getsize(temp_file) / (1024 * 1024)  # MB
                        os.remove(temp_file)
                        logger.logger.info(f"✅ Удален временный файл: {temp_file} ({file_size:.2f} MB)")
                    except Exception as e:
                        logger.logger.warning(f"⚠️ Не удалось удалить временный файл {temp_file}: {e}")
                else:
                    logger.logger.debug(f"Временный файл уже не существует: {temp_file}")

            if caption_success:
                logger.logger.info("✅ Субтитры добавлены успешно")

                # Проверяем финальный результат
                if not os.path.exists(final_output):
                    logger.logger.error(f"❌ Финальный файл шорта не найден: {final_output}")
                    return None
                else:
                    file_size = os.path.getsize(final_output) / (1024 * 1024)  # MB
                    logger.logger.info(f"✅ Финальный шорт создан: {final_output} ({file_size:.2f} MB)")

                    # Проверяем, что файл не пустой
                    if file_size < 0.1:  # менее 100KB
                        logger.logger.error(f"❌ Финальный файл слишком маленький: {file_size:.2f} MB")
                        try:
                            os.remove(final_output)
                            logger.logger.info(f"Удален пустой финальный файл: {final_output}")
                        except Exception as e:
                            logger.logger.warning(f"Не удалось удалить пустой файл: {e}")
                        return None

                # Сохраняем информацию в БД
                logger.logger.info("Сохранение информации в базу данных...")
                try:
                    ctx.db.add_highlight(
                        ctx.video_id,
                        start,
                        adjusted_stop,
                        final_output,
                        segment_text=highlight_item.get('segment_text', ''),
                        caption_with_hashtags=highlight_item.get('caption_with_hashtags', '')
                    )
                    logger.logger.info("✅ Информация успешно сохранена в БД")
                except Exception as e:
                    logger.logger.warning(f"⚠️ Не удалось сохранить в БД: {e}")
                    import traceback
                    logger.logger.warning(f"Traceback: {traceback.format_exc()}")

                logger.logger.info(f"--- МОМЕНТ {seq} ОБРАБОТАН УСПЕШНО ---")
                return final_output
            else:
                logger.logger.error(f"❌ Шаг 3 ПРОВАЛЕН: Не удалось добавить субтитры для момента {seq}")
                return None

        except Exception as e:
            logger.logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА при обработке момента {seq}: {e}")
            import traceback
            logger.logger.error(f"Traceback: {traceback.format_exc()}")

            # Экстренная очистка временных файлов при ошибке
            logger.logger.info("Экстренная очистка временных файлов после ошибки...")
            temp_files_to_clean = [temp_segment, cropped_vertical, final_output]
            for temp_file in temp_files_to_clean:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        logger.logger.info(f"Удален файл после ошибки: {temp_file}")
                    except Exception as clean_e:
                        logger.logger.warning(f"Не удалось удалить файл после ошибки {temp_file}: {clean_e}")

            return None

    def _create_result(self, video_path: str, ranked_moments: List[RankedMoment], transcription_data: Dict[str, Any], generated_shorts: List[str] = None) -> FilmAnalysisResult:
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
        if generated_shorts is None:
            generated_shorts = []

        metadata = {
            'processed_at': datetime.now().isoformat(),
            'model_version': self.config.llm.model_name,
            'total_segments_analyzed': len(transcription_data.get('segments', [])),
            'video_duration': duration,
            'moments_found': len(ranked_moments),
            'shorts_generated': len(generated_shorts),
            'film_config': {
                'min_quality_score': self.film_config.min_quality_score,
                'generate_shorts': self.film_config.generate_shorts,
                'pause_threshold': self.film_config.pause_threshold
            }
        }

        return FilmAnalysisResult(
            video_id=video_id,
            duration=duration,
            keep_ranges=keep_ranges,
            scores=scores,
            preview_text=preview_text,
            risks=risks,
            metadata=metadata,
            generated_shorts=generated_shorts
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
            metadata={'error': True},
            generated_shorts=[]
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