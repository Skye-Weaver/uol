"""
Режим "фильм" для анализа видео и выделения лучших моментов.
Анализирует длинные видео и предлагает оптимальные фрагменты для создания фильма из лучших частей.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import json
import os
from datetime import datetime

from Components.YoutubeDownloader import download_youtube_video
from Components.Transcription import transcribe_unified
from Components.LanguageTasks import (
    build_transcription_prompt,
    GetHighlights,
    call_llm_with_retry,
    call_llm_with_film_mode_retry,
    make_generation_config,
    compute_tone_and_keywords,
    compute_emojis_for_segment
)
from Components.Transcription import prepare_words_for_segment
from Components.Database import VideoDatabase
from Components.config import get_config, AppConfig
from Components.Logger import logger
from Components.Edit import crop_video, burn_captions, crop_bottom_video, animate_captions, get_video_dimensions
from Components.FaceCrop import crop_to_70_percent_with_blur, crop_to_vertical_average_face
from Components.Paths import build_short_output_name
from faster_whisper import WhisperModel
import math
import subprocess


@dataclass
class FilmMoment:
    """Структура для хранения информации о моменте фильма"""
    moment_type: str  # "COMBO" или "SINGLE"
    start_time: float
    end_time: float
    text: str
    segments: List[Dict[str, Any]] = field(default_factory=list)  # Для COMBO: суб-сегменты
    context: str = ""  # Описание контекста
    keywords: List[str] = field(default_factory=list)  # Ключевые слова, выделенные ИИ


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
        self._last_ranking_info = {}
        # Контекст транскрипции для расширенного скоринга (pace/silence)
        self._ctx_transcription_data = None

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
        try:
            video_path, transcription_data = self._get_video_and_transcription(url, local_path)
            if not video_path or not transcription_data:
                logger.logger.error("Не удалось получить видео или транскрибацию")
                return self._create_empty_result("")
        except Exception as e:
            logger.logger.error(f"Ошибка при получении видео и транскрибации: {e}")
            return self._create_empty_result("")

        # Логируем информацию о полученных данных + единое разрешение длительности
        segments_count = len(transcription_data.get('segments', []))
        # Единое разрешение длительности и запись в transcription_data['duration']
        resolved_duration = self._resolve_video_duration(video_path, transcription_data)
        logger.logger.info(f"Получены данные транскрибации: сегментов={segments_count}")
        logger.logger.info(f"Duration: {resolved_duration:.2f} seconds")
        # Сохраняем транскрипцию в контекст для последующих фаз (скоринг pace/silence)
        try:
            self._ctx_transcription_data = transcription_data
        except Exception:
            self._ctx_transcription_data = None

        # 2. Анализ моментов через ИИ
        moments = self._analyze_moments(transcription_data)
        if not moments:
            logger.logger.warning("Не найдено подходящих моментов для анализа")
            logger.logger.warning(f"Данные транскрибации: duration={resolved_duration}, segments={segments_count}")
            return self._create_empty_result(video_path)

        # 3. Ранжирование моментов
        ranked_moments = self._rank_moments(moments)
        try:
            info = getattr(self, "_last_ranking_info", {}) or {}
            strategy = info.get("selection_strategy", "quality_threshold")
            logger.logger.info(f"[OK] Ranking: selected={len(ranked_moments)} (strategy={strategy}), proceed")
        except Exception:
            pass
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
        try:
            _ctx_dur = float(transcription_data.get('duration', 0.0) or 0.0)
            logger.logger.info(f"[OK] Duration consistency: ctx={_ctx_dur:.2f}s, warnings=0")
            logger.logger.info(f"[OK] Summary duration: {_ctx_dur:.2f}s записано в JSON и в финальный лог")
        except Exception:
            logger.logger.info(f"[OK] Duration consistency: ctx={transcription_data.get('duration', 0.0)}s, warnings=0")
            logger.logger.info(f"[OK] Summary duration: {transcription_data.get('duration', 0.0)}s записано в JSON и в финальный лог")
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

    def _resolve_video_duration(self, video_path: str, transcription_data: Dict[str, Any]) -> float:
        """
        Единое определение длительности видео с приоритетом:
          a) транскрипция (макс. end среди сегментов word-level и/или legacy)
          b) fallback: ffprobe через _get_video_duration
        Валидация: значение > 0; логируем INFO итог; при сомнении — ERROR и переключение на альтернативный источник.
        """
        # Извлекаем длительность из транскрипции
        trans_end_word = 0.0
        try:
            wl = transcription_data.get('word_level', {})
            if isinstance(wl, dict) and isinstance(wl.get('segments'), list) and wl['segments']:
                trans_end_word = max(float(s.get('end', 0.0) or 0.0) for s in wl['segments'])
        except Exception as e:
            logger.logger.debug(f"Не удалось вычислить длительность из word_level: {e}")

        trans_end_legacy = 0.0
        try:
            segs = transcription_data.get('segments') or []
            ends = []
            for seg in segs:
                if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                    ends.append(float(seg[2]))
                elif isinstance(seg, dict):
                    ends.append(float(seg.get('end', 0.0) or 0.0))
            if ends:
                trans_end_legacy = max(ends)
        except Exception as e:
            logger.logger.debug(f"Не удалось вычислить длительность из legacy segments: {e}")

        transcription_duration = max(trans_end_word, trans_end_legacy)

        # Fallback: ffprobe
        ffprobe_duration = 0.0
        try:
            ffprobe_duration = float(self._get_video_duration(video_path) or 0.0)
        except Exception as e:
            logger.logger.debug(f"Не удалось получить длительность через ffprobe: {e}")
            ffprobe_duration = 0.0

        # Сохраняем источники в transcription_data для последующего логирования
        transcription_data['duration_from_transcription'] = float(transcription_duration or 0.0)
        transcription_data['duration_from_ffprobe'] = float(ffprobe_duration or 0.0)

        # Выбор значения по правилам
        chosen = transcription_duration if transcription_duration and transcription_duration > 0 else ffprobe_duration

        # Диагностика сомнительных случаев
        long_transcription = transcription_duration and transcription_duration > 600
        if (not chosen or chosen <= 0):
            logger.logger.error("[DURATION] Получено некорректное значение длительности (<= 0). Переключение на альтернативный источник (ffprobe).")
            if ffprobe_duration and ffprobe_duration > 0:
                chosen = ffprobe_duration

        if long_transcription and (chosen < 300 or chosen <= 0):
            logger.logger.error("[DURATION] Подозрительно малая длительность при длинной транскрипции. Переключение на альтернативный источник.")
            alt = ffprobe_duration if chosen == transcription_duration else transcription_duration
            if alt and alt > chosen:
                chosen = alt

        # Если оба источника валидны и сильно расходятся, выбираем большее для предотвращения обрезки
        try:
            if transcription_duration > 0 and ffprobe_duration > 0:
                delta = abs(transcription_duration - ffprobe_duration)
                if delta > 1.0 and ffprobe_duration > transcription_duration * 1.05:
                    logger.logger.warning(f"[DURATION] Расхождение источников: transcription={transcription_duration:.2f}s, ffprobe={ffprobe_duration:.2f}s. Выбрано большее значение.")
                    chosen = max(transcription_duration, ffprobe_duration)
        except Exception:
            pass

        # Финальные логи
        if chosen and chosen > 0:
            logger.logger.info(f"[VALIDATION] duration sources: transcription={transcription_duration:.2f}, ffprobe={ffprobe_duration:.2f}, chosen=ctx.video_duration={chosen:.2f}")
            logger.logger.info(f"Duration: {chosen:.2f} seconds")
        else:
            logger.logger.error("[DURATION] Не удалось надежно определить длительность видео. Установлено 0.0s")
            chosen = 0.0

        # Записываем единое значение в transcription_data
        transcription_data['duration'] = float(chosen)
        return float(chosen)

    def _analyze_moments(self, transcription_data: Dict[str, Any]) -> List[FilmMoment]:
        """Анализ моментов через ИИ (Film Mode v2: оконный сбор для длинных фильмов)"""
        try:
            duration = float(transcription_data.get('duration', 0.0) or 0.0)
        except Exception:
            duration = 0.0

        try:
            # АКТИВИРУЕМ ОКОННЫЙ РЕЖИМ ДЛЯ ВСЕХ ФИЛЬМОВ > 10 МИНУТ (было 45 минут)
            if duration >= 10 * 60:  # 10 минут вместо 45
                logger.logger.info(f"[WINDOW] Активирован оконный режим извлечения кандидатов (duration={duration:.2f}s)")
                try:
                    moments = self._extract_film_moments_windowed(transcription_data)
                    if moments is None:
                        logger.logger.warning("[WINDOW] Не удалось извлечь моменты в оконном режиме")
                        return []
                    logger.logger.info(f"[WINDOW] Найдено {len(moments)} кандидатов после дедупликации")
                    return moments
                except Exception as e:
                    logger.logger.error(f"Ошибка в оконном режиме анализа моментов: {e}")
                    return []

            # Короткие видео: прежняя монолитная логика
            segments_legacy = transcription_data.get('segments', [])
            if not segments_legacy:
                logger.logger.warning("Отсутствуют сегменты транскрибации для анализа")
                return []

            segments_dict = []
            for seg in segments_legacy:
                try:
                    if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                        segments_dict.append({
                            'text': str(seg[0]),
                            'start': float(seg[1]),
                            'end': float(seg[2])
                        })
                    elif isinstance(seg, dict):
                        segments_dict.append(seg)
                except Exception as e:
                    logger.logger.debug(f"Ошибка при обработке сегмента: {e}, пропускаем")
                    continue

            if not segments_dict:
                logger.logger.warning("Не удалось преобразовать сегменты транскрибации")
                return []

            logger.logger.info("Формирование текста транскрибации через build_transcription_prompt...")
            try:
                transcription_text = build_transcription_prompt(segments_dict)
                logger.logger.info(f"✅ Текст транскрибации сформирован: {len(transcription_text)} символов")
            except Exception as e:
                logger.logger.error(f"Ошибка при формировании текста транскрибации: {e}")
                return []

            logger.logger.info("Анализ моментов через LLM (монолитный вызов)...")
            try:
                moments = self._extract_film_moments(transcription_text)
                if moments is None:
                    logger.logger.warning("Не удалось извлечь моменты из транскрибации")
                    return []
                logger.logger.info(f"Найдено {len(moments)} потенциальных моментов")
                return moments
            except Exception as e:
                logger.logger.error(f"Ошибка при извлечении моментов через LLM: {e}")
                return []
        except Exception as e:
            logger.logger.error(f"Критическая ошибка при анализе моментов: {e}")
            return []

    def _extract_film_moments(self, transcription: str) -> List[FilmMoment]:
        """Извлечение моментов фильма через LLM с ключевыми словами"""
        system_instruction = f"""
        Ты — эксперт по анализу видео контента для создания вирусных shorts. Проанализируй предоставленную транскрибацию и выдели лучшие моменты двух типов:

        1. COMBO (10-20 сек): Склейка 2-4 коротких кусков из одной сцены в хронологическом порядке для создания мини-дуги
        2. SINGLE (30-60 сек): Один самодостаточный момент с микро-аркой (завязка → нарастание → развязка)

        Для КАЖДОГО момента выдели 3-7 ключевых слов, которые характеризуют его суть и потенциал для вирусности.

        Верни ТОЛЬКО JSON-массив объектов с полями:
        - moment_type: "COMBO" или "SINGLE"
        - start_time: число (секунды)
        - end_time: число (секунды)
        - text: текст момента
        - context: краткое описание почему этот момент подходит
        - keywords: массив строк с ключевыми словами (3-7 слов)

        Для COMBO также добавь:
        - segments: массив суб-сегментов с start/end/text

        Найди до {max(self.film_config.max_moments, self.film_config.target_shorts_count)} лучших моментов.
        """

        try:
            logger.logger.info(f"Отправка запроса к LLM для анализа моментов (модель: {self.film_config.llm_model})")
            logger.logger.debug(f"Длина транскрибации: {len(transcription)} символов")

            logger.logger.info("Создание конфигурации генерации через make_generation_config...")
            generation_config = make_generation_config(system_instruction, temperature=0.3)
            logger.logger.info("✅ Конфигурация генерации создана")

            logger.logger.info("Отправка запроса к LLM через call_llm_with_film_mode_retry...")
            response = call_llm_with_film_mode_retry(
                system_instruction=None,
                content=transcription,
                generation_config=generation_config,
                model=self.film_config.llm_model,
                max_api_attempts=5,
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
                        segments=item.get('segments', []),
                        keywords=item.get('keywords', [])
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
        """Ранжирование моментов с динамическим порогом и покрытием по таймлайнам"""
        # Конфигурация ранжирования
        rc = getattr(self.film_config, "ranking", {}) or {}
        min_thr_cfg = float(rc.get("min_quality_threshold", getattr(self.film_config, "min_quality_score", 0.5)))
        soft_min = float(rc.get("soft_min_quality", 0.35))
        allow_fb = bool(rc.get("allow_fallback", True))
        fb_top_n = int(max(1, rc.get("fallback_top_n", 12)))
        max_best_cfg = int(max(1, rc.get("max_best_moments", 30)))
        target_n = int(max(1, getattr(self.film_config, "target_shorts_count", max_best_cfg)))
        bucket_min = int(max(1, getattr(self.film_config, "diversity_bucket_minutes", 5)))
        bucket_sec = bucket_min * 60
        gen_top_k = int(max(1, getattr(self.film_config, "generator_top_k", target_n)))

        # Подсчет и сортировка по score (с учетом ключевых слов)
        scored: List[RankedMoment] = []
        for moment in moments or []:
            scores = self._calculate_moment_scores(moment)
            total_score = sum(
                scores.get(name, 0.0) * self.film_config.ranking_weights.get(name, 0.0)
                for name in self.film_config.ranking_weights.keys()
            )
            # Бонус за количество и качество ключевых слов
            keyword_bonus = len(moment.keywords or []) * 0.1
            total_score += keyword_bonus
            scored.append(RankedMoment(moment=moment, scores=scores, total_score=total_score, rank=0))

        scored.sort(key=lambda x: x.total_score, reverse=True)
        M = len(scored)
        logger.logger.info(f"Кандидатов: {M}, min_thr_cfg={min_thr_cfg}, soft={soft_min}, target_n={target_n}, max_best_cfg={max_best_cfg}, generator_top_k={gen_top_k}")

        if M == 0:
            return []

        # Динамический порог: p75
        totals = [rm.total_score for rm in scored]
        totals_sorted = sorted(totals)
        try:
            p75_idx = max(0, min(len(totals_sorted) - 1, int(round(0.75 * (len(totals_sorted) - 1)))))
            p75 = float(totals_sorted[p75_idx])
        except Exception:
            p75 = min_thr_cfg
        min_thr_dyn = max(min_thr_cfg, p75)
        logger.logger.info(f"[RANK] dynamic threshold p75={p75:.3f} -> min_thr_dyn={min_thr_dyn:.3f}")

        selected = [rm for rm in scored if rm.total_score >= min_thr_dyn]

        strategy = "dynamic_p75"
        if not selected and allow_fb:
            # Fallback: берем top-N
            K = max(1, fb_top_n)
            selected = scored[:K]
            strategy = "fallback_topN"
            logger.logger.warning(f"Fallback top-N активирован: выбранных={len(selected)} из {M}")

        # Ограничение предварительное
        if selected:
            selected = selected[:max_best_cfg]

        # Покрытие по таймлайнам (diversity buckets) с round-robin
        if selected:
            buckets: Dict[int, List[RankedMoment]] = {}
            for rm in selected:
                try:
                    b = int(max(0, rm.moment.start_time) // bucket_sec)
                except Exception:
                    b = 0
                buckets.setdefault(b, []).append(rm)

            # внутри каждого бакета уже по убыванию total_score
            for b in buckets.values():
                b.sort(key=lambda x: x.total_score, reverse=True)

            covered: List[RankedMoment] = []
            keys = sorted(buckets.keys())
            # Итоговый лимит
            limit_k = min(gen_top_k, target_n, max_best_cfg, len(selected))
            idx = 0
            while len(covered) < limit_k:
                progressed = False
                for k in keys:
                    bucket_list = buckets.get(k, [])
                    if idx < len(bucket_list):
                        covered.append(bucket_list[idx])
                        progressed = True
                        if len(covered) >= limit_k:
                            break
                if not progressed:
                    # все бакеты исчерпаны на данном idx
                    break
                idx += 1
            selected = covered

        # Присвоение рангов
        for i, rm in enumerate(selected):
            rm.rank = i + 1

        first_score = selected[0].total_score if selected else 0.0
        last_score = selected[-1].total_score if selected else 0.0
        logger.logger.info(f"Выбрано {len(selected)} моментов (strategy={strategy}), первый score={first_score:.3f}, последний score={last_score:.3f}")
        logger.logger.info(f"[OK] Ranking: candidates={M}, dynamic_thr={min_thr_dyn:.3f}, selected={len(selected)} (strategy={strategy})")
        if len(selected) >= 1:
            logger.logger.info("[VALIDATION] Ranking produced N>=1: OK")

        try:
            self._last_ranking_info = {
                "selection_strategy": strategy,
                "candidates": M,
                "selected": len(selected),
                "threshold": float(min_thr_dyn),
                "soft": float(soft_min),
            }
        except Exception:
            pass

        return selected

    def _calculate_moment_scores(self, moment: FilmMoment) -> Dict[str, float]:
        """Расчет оценок момента по совпадениям ключевых слов (новая система ранжирования)"""
        scores: Dict[str, float] = {}

        # Получаем эталонные ключевые слова из конфигурации
        ref_keywords = getattr(self.film_config, 'reference_keywords', {})

        # Нормализуем ключевые слова момента (приводим к нижнему регистру)
        moment_keywords = [kw.lower().strip() for kw in (moment.keywords or []) if kw and kw.strip()]

        # Подсчитываем совпадения для каждой категории
        for category, reference_words in ref_keywords.items():
            if category in self.film_config.ranking_weights:
                # Нормализуем эталонные ключевые слова
                ref_words_lower = [w.lower().strip() for w in reference_words]

                # Считаем совпадения
                matches = 0
                for moment_kw in moment_keywords:
                    # Проверяем точное совпадение или частичное вхождение
                    for ref_kw in ref_words_lower:
                        if ref_kw in moment_kw or moment_kw in ref_kw:
                            matches += 1
                            break  # Одно ключевое слово момента может соответствовать только одной категории

                # Нормализуем оценку (максимум 10 за совпадения)
                scores[category] = min(matches * 2.0, 10.0)

        # Длина текста как бонус (короткие моменты лучше для shorts)
        duration = moment.end_time - moment.start_time
        if 10 <= duration <= 60:  # Идеальная длительность для shorts
            scores['duration_bonus'] = 4.0
        elif 5 <= duration <= 120:  # Приемлемая длительность
            scores['duration_bonus'] = 2.0
        else:
            scores['duration_bonus'] = 0.0

        # Бонус за разнообразие типов моментов
        if moment.moment_type == 'COMBO':
            scores['combo_bonus'] = 3.0  # COMBO моменты более ценны
        else:
            scores['combo_bonus'] = 0.0

        # Базовый скор за наличие ключевых слов
        if moment_keywords:
            scores['content_bonus'] = min(len(moment_keywords) * 0.5, 3.0)
        else:
            scores['content_bonus'] = 0.0

        # Штраф за визуальную зависимость
        text = (moment.text or "").lower()
        visual_keywords = ['визуально', 'зрительно', 'видно', 'картинка', 'изображение']
        visual_count = sum(1 for keyword in visual_keywords if keyword in text)
        scores['visual_penalty'] = -visual_count * 0.2

        # Расширенные метрики: pace/silence из контекста транскрипции
        try:
            td = getattr(self, "_ctx_transcription_data", None) or {}
            pace, sil = self._compute_pace_silence_scores(moment, td)
            scores['pace_score'] = max(0.0, min(10.0, float(pace)))
            scores['silence_penalty'] = max(0.0, min(10.0, float(sil)))
        except Exception:
            scores.setdefault('pace_score', 0.0)
            scores.setdefault('silence_penalty', 0.0)

        # Финальная нормализация к шкале 0-10
        for key in list(scores.keys()):
            try:
                scores[key] = min(max(float(scores[key]), -2), 10)  # Разрешаем небольшие отрицательные значения
            except Exception:
                scores[key] = 0.0

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
        """Интеллектуальное обнаружение скучных сегментов в моменте с использованием ИИ-анализа пауз"""
        try:
            # Импортируем анализатор пауз
            from Components.PauseAnalysis import analyze_pauses_in_moment

            # Анализируем паузы в рамках момента
            pause_analyses = analyze_pauses_in_moment(
                moment.start_time,
                moment.end_time,
                transcription_data
            )

            boring_segments = []

            # Преобразуем результаты анализа пауз в формат boring_segments
            for pause_analysis in pause_analyses:
                if pause_analysis.should_trim:
                    boring_segments.append({
                        'start': pause_analysis.start_time,
                        'end': pause_analysis.end_time,
                        'reason': f'intelligent_{pause_analysis.category}',
                        'confidence': pause_analysis.confidence,
                        'importance_score': pause_analysis.importance_score,
                        'should_trim': pause_analysis.should_trim,
                        'ai_reasoning': pause_analysis.reasoning,
                        'duration': pause_analysis.duration
                    })

            # Если ИИ-анализ не дал результатов или отключен, используем легаси-метод
            if not boring_segments and not getattr(self.film_config, 'intelligent_pause_analysis', {}).get('enabled', False):
                logger.logger.info("ИИ-анализ пауз отключен или не дал результатов, используем легаси-метод")
                boring_segments = self._detect_boring_segments_legacy(moment, transcription_data)

            logger.logger.debug(f"Обнаружено {len(boring_segments)} скучных сегментов в моменте "
                              f"({moment.start_time:.1f}s-{moment.end_time:.1f}s)")

            return boring_segments

        except Exception as e:
            logger.logger.error(f"Ошибка при интеллектуальном анализе пауз: {e}. "
                               "Откатываемся на легаси-метод.")
            # Fallback на легаси-метод при ошибках
            return self._detect_boring_segments_legacy(moment, transcription_data)

    def _detect_boring_segments_legacy(self, moment: FilmMoment, transcription_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Старая логика обнаружения скучных сегментов (резервная)"""
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
        """Интеллектуальное применение обрезки к моменту с учетом ИИ-анализа пауз"""
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
        # Добавляем информацию об ИИ-анализе пауз в контекст
        ai_trimmed_count = sum(1 for seg in boring_segments if seg.get('reason', '').startswith('intelligent_'))
        legacy_trimmed_count = len(boring_segments) - ai_trimmed_count

        context_parts = [moment.context]
        if ai_trimmed_count > 0:
            context_parts.append(f"ИИ-обрезка: {ai_trimmed_count} пауз")
        if legacy_trimmed_count > 0:
            context_parts.append(f"Стандартная обрезка: {legacy_trimmed_count} пауз")

        return FilmMoment(
            moment_type=moment.moment_type,
            start_time=new_segments[0]['start'],
            end_time=new_segments[-1]['end'],
            text=moment.text,
            segments=moment.segments,
            context=f" ({'; '.join(context_parts)})"
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
        # Валидация источников длительности для шортов
        try:
            _td = float(transcription_data.get('duration_from_transcription', 0.0) or 0.0)
            _fd = float(transcription_data.get('duration_from_ffprobe', 0.0) or 0.0)
            _ch = float(transcription_data.get('duration', 0.0) or 0.0)
            logger.logger.info(f"[VALIDATION] duration sources: transcription={_td:.2f}, ffprobe={_fd:.2f}, chosen=ctx.video_duration={_ch:.2f}")
        except Exception:
            pass

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
        top_k = int(max(1, getattr(self.film_config, "generator_top_k", getattr(self.film_config, "target_shorts_count", 30))))
        top_moments = ranked_moments[:top_k]
        logger.logger.info(f"Выбрано топ-{len(top_moments)} моментов из {len(ranked_moments)} (максимум {top_k})")

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

                # Генерируем шорт
                logger.logger.info(f"Вызов _process_moment_to_short для момента {rm.rank}")
                short_path = self._process_moment_to_short(processing_context, moment, i + 1)

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
                else:
                    # Валидация суб-сегментов
                    for i, seg in enumerate(moment.segments):
                        if not isinstance(seg, dict):
                            issues.append(f"Суб-сегмент {i} не является словарем")
                            is_valid = False
                            continue
                        seg_start = seg.get('start')
                        seg_end = seg.get('end')
                        if seg_start is None or seg_end is None:
                            issues.append(f"Суб-сегмент {i} без start/end")
                            is_valid = False
                        elif seg_end <= seg_start:
                            issues.append(f"Суб-сегмент {i} имеет некорректный интервал: {seg_start} >= {seg_end}")
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
                try:
                    # Обработка нормализованного формата [text, start, end]
                    if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                        seg_start = float(seg[1])
                        seg_end = float(seg[2])
                        # Проверяем пересечение с моментом
                        if seg_end > start_time and seg_start < end_time:
                            relevant_segments.append(seg)
                    # Обработка dict формата (если вдруг попал)
                    elif isinstance(seg, dict):
                        seg_start = float(seg.get("start", 0.0))
                        seg_end = float(seg.get("end", 0.0))
                        if seg_end > start_time and seg_start < end_time:
                            # Конвертируем в список для совместимости
                            text = str(seg.get("text", ""))
                            relevant_segments.append([text, seg_start, seg_end])
                    else:
                        logger.logger.debug(f"Пропускаем сегмент неизвестного формата: {type(seg)}")
                except Exception as e:
                    logger.logger.debug(f"Ошибка при обработке сегмента: {e}, пропускаем")
                    continue

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

    def _get_video_duration_from_transcription(self, ctx) -> Optional[float]:
        """Получение длительности видео из данных транскрибации с улучшенной логикой"""
        try:
            # 1. Сначала пробуем получить из word_level_transcription
            if ctx.word_level_transcription and 'segments' in ctx.word_level_transcription:
                segments = ctx.word_level_transcription['segments']
                if segments:
                    # Берем время окончания последнего сегмента
                    last_segment = segments[-1]
                    if isinstance(last_segment, dict) and 'end' in last_segment:
                        duration = float(last_segment['end'])
                        logger.logger.debug(f"Длительность из word_level_transcription: {duration:.2f}s")
                        return duration

            # 2. Пробуем из обычных segments
            if ctx.transcription_segments:
                # Ищем последний сегмент с корректными данными
                for seg in reversed(ctx.transcription_segments):
                    if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                        try:
                            end_time = float(seg[2])
                            if end_time > 0:
                                logger.logger.debug(f"Длительность из transcription_segments: {end_time:.2f}s")
                                return end_time
                        except (ValueError, IndexError):
                            continue

            # 3. Fallback: пытаемся получить через ffprobe
            try:
                duration = self._get_video_duration(ctx.video_path)
                if duration and duration > 0:
                    logger.logger.debug(f"Длительность через ffprobe: {duration:.2f}s")
                    return duration
            except Exception as e:
                logger.logger.warning(f"Не удалось получить длительность через ffprobe: {e}")

            logger.logger.warning("Не удалось определить длительность видео из доступных источников")
            return None

        except Exception as e:
            logger.logger.error(f"Ошибка при определении длительности видео: {e}")
            return None

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
                # Единая длительность видео для всего пайплайна
                try:
                    self.video_duration = float(transcription_data.get('duration', 0.0) or 0.0)
                except Exception:
                    self.video_duration = 0.0
                # Источники длительности (для диагностики)
                try:
                    self.duration_from_transcription = float(transcription_data.get('duration_from_transcription', 0.0) or 0.0)
                except Exception:
                    self.duration_from_transcription = 0.0
                try:
                    self.duration_from_ffprobe = float(transcription_data.get('duration_from_ffprobe', 0.0) or 0.0)
                except Exception:
                    self.duration_from_ffprobe = 0.0

        return MockProcessingContext(video_path, video_id, transcription_data, width, height, self.config)

    def _process_moment_to_short(self, ctx, moment: FilmMoment, seq: int) -> Optional[str]:
        """Обработка момента в шорт с поддержкой склеивания суб-сегментов для COMBO"""
        logger.logger.info(f"--- НАЧАЛО ОБРАБОТКИ МОМЕНТА {seq} ---")

        final_output = None
        temp_segments = []
        cropped_verticals = []

        try:
            # 1. Определение типа момента и таймкодов
            if moment.moment_type == 'COMBO' and moment.segments:
                # Для COMBO обрабатываем суб-сегменты
                logger.logger.info(f"Обработка COMBO момента с {len(moment.segments)} суб-сегментами")
                segment_outputs = []

                for sub_idx, sub_seg in enumerate(moment.segments):
                    try:
                        sub_start = float(sub_seg.get('start', 0))
                        sub_end = float(sub_seg.get('end', 0))
                        sub_text = sub_seg.get('text', '')

                        if sub_end <= sub_start:
                            logger.logger.warning(f"Пропуск некорректного суб-сегмента {sub_idx}: {sub_start} >= {sub_end}")
                            continue

                        # Обработка суб-сегмента
                        sub_output = self._process_single_segment(ctx, sub_start, sub_end, sub_text, seq, sub_idx)
                        if sub_output:
                            segment_outputs.append(sub_output)
                        else:
                            logger.logger.warning(f"Не удалось обработать суб-сегмент {sub_idx}")

                    except Exception as e:
                        logger.logger.warning(f"Ошибка при обработке суб-сегмента {sub_idx}: {e}")
                        continue

                if not segment_outputs:
                    logger.logger.error(f"❌ Не удалось обработать ни один суб-сегмент для COMBO момента {seq}")
                    return None

                # Склеивание суб-сегментов
                final_output = self._concatenate_segments(segment_outputs, seq, ctx)
                if not final_output:
                    logger.logger.error(f"❌ Не удалось склеить суб-сегменты для момента {seq}")
                    return None

            else:
                # Для SINGLE обрабатываем как один сегмент
                start = float(moment.start_time)
                stop = float(moment.end_time)
                adjusted_stop = math.ceil(stop)
                logger.logger.info(f"Таймкоды SINGLE: {start:.2f}s - {adjusted_stop:.2f}s (округлено с {stop:.2f}s)")

                if adjusted_stop <= start:
                    logger.logger.error(f"❌ Некорректные таймкоды: start={start} >= end={adjusted_stop}")
                    return None

                final_output = self._process_single_segment(ctx, start, adjusted_stop, moment.text, seq, 0)
                if not final_output:
                    logger.logger.error(f"❌ Не удалось обработать SINGLE момент {seq}")
                    return None

            # 2. Определение путей файлов
            base_name = os.path.splitext(os.path.basename(ctx.video_path))[0]
            # Используем централизованную функцию для имен
            final_output, _ = build_short_output_name(base_name, seq, ctx.cfg.processing.shorts_dir)
            
            output_base = f"{base_name}_film_moment_{seq}"
            temp_segment = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_temp.mp4")
            cropped_vertical = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_vertical.mp4")

            os.makedirs(ctx.cfg.processing.videos_dir, exist_ok=True)
            os.makedirs(ctx.cfg.processing.shorts_dir, exist_ok=True)

            logger.logger.info("Пути файлов:")
            logger.logger.info(f"  Временный сегмент: {temp_segment}")
            logger.logger.info(f"  Вертикальный кроп: {cropped_vertical}")
            logger.logger.info(f"  Финальный шорт: {final_output}")

            # --- ШАГ 1: Извлечение сегмента видео ---
            extract_success = self._extract_video_segment(
                ctx.video_path, temp_segment, start, adjusted_stop, ctx.initial_width, ctx.initial_height
            )
            if not extract_success:
                logger.logger.error(f"❌ Не удалось извлечь сегмент для момента {seq}")
                self._cleanup_temp_files([temp_segment], f" после неудачного извлечения сегмента {seq}")
                return None

            # --- ШАГ 2: Создание вертикального кропа ---
            crop_mode = ctx.cfg.processing.crop_mode
            crop_function = crop_to_70_percent_with_blur if crop_mode == "70_percent_blur" else crop_to_vertical_average_face
            
            vert_crop_path = self._safe_file_operation(
                f"создание вертикального кропа для момента {seq}",
                crop_function,
                temp_segment, cropped_vertical
            )
            if not vert_crop_path:
                logger.logger.error(f"❌ Не удалось создать вертикальный кроп для момента {seq}")
                self._cleanup_temp_files([temp_segment], f" после неудачного кропа {seq}")
                return None
            
            # --- ШАГ 3: ДОБАВЛЕНИЕ СУБТИТРОВ (ИСПРАВЛЕНО) ---
            captioning_success = False
            use_animated = ctx.cfg.processing.use_animated_captions

            logger.logger.info(f"Режим субтитров: {'Анимированные' if use_animated else 'Статичные (ASS)'}")

            if use_animated:
                # Логика для анимированных субтитров
                # Подготовка данных на уровне слов для сегмента
                transcription_result = prepare_words_for_segment(
                    ctx.word_level_transcription, start, adjusted_stop
                )

                if transcription_result and transcription_result.get("segments"):
                    # Получение метаданных для выделения (тон, ключевые слова, эмодзи)
                    segment_text = highlight_item.get('segment_text', '')
                    meta = compute_tone_and_keywords(segment_text) if segment_text else {}
                    
                    cfg_emoji = getattr(ctx.cfg.captions, "emoji", None)
                    if cfg_emoji and getattr(cfg_emoji, "enabled", False) and segment_text:
                        tone_val = meta.get("tone", "neutral")
                        max_per = int(getattr(cfg_emoji, "max_per_short", 0) or 0)
                        emojis = compute_emojis_for_segment(segment_text, tone_val, max_per)
                        meta["emojis"] = emojis

                    captioning_success = animate_captions(
                        cropped_vertical,
                        temp_segment,  # Источник аудио
                        transcription_result,
                        final_output,
                        style_cfg=ctx.cfg.captions,
                        highlight_meta=meta
                    )
                else:
                    logger.logger.warning("Нет данных на уровне слов для анимации, пропуск.")

            else:
                # Логика для статичных субтитров (ASS)
                captioning_success = burn_captions(
                    cropped_vertical,
                    temp_segment, # Источник аудио
                    ctx.transcription_segments,
                    start,
                    adjusted_stop,
                    final_output,
                    style_cfg=ctx.cfg.captions
                )
            
            # --- ФИНАЛИЗАЦИЯ ---
            if captioning_success:
                logger.logger.info(f"✅ Успешно обработан момент {seq}. Финальный файл: {final_output}")
                ctx.db.add_highlight(
                    ctx.video_id, start, adjusted_stop, final_output,
                    segment_text=highlight_item.get('segment_text', ''),
                    caption_with_hashtags=highlight_item.get('caption_with_hashtags', '')
                )
                return final_output
            else:
                logger.logger.error(f"❌ Не удалось добавить субтитры для момента {seq}")
                return None

            # Финализация
            if final_output:
                logger.logger.info(f"✅ Успешно обработан момент {seq}. Финальный файл: {final_output}")
                ctx.db.add_highlight(
                    ctx.video_id, moment.start_time, moment.end_time, final_output,
                    segment_text=moment.text,
                    caption_with_hashtags=f"Film Moment {seq}: {moment.text[:100]}..."
                )
                return final_output
            else:
                logger.logger.error(f"❌ Не удалось обработать момент {seq}")
                return None

        except Exception as e:
            logger.logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА при обработке момента {seq}: {e}")
            import traceback
            logger.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
            # Очистка временных файлов
            self._cleanup_temp_files(temp_segments + cropped_verticals, f" после обработки момента {seq}")

    def _process_single_segment(self, ctx, start: float, end: float, text: str, seq: int, sub_idx: int = 0) -> Optional[str]:
        """Обработка одного сегмента видео (извлечение, кроп, субтитры)"""
        temp_segment = None
        cropped_vertical = None

        try:
            # 2. Определение путей файлов
            base_name = os.path.splitext(os.path.basename(ctx.video_path))[0]
            output_base = f"{base_name}_film_moment_{seq}"
            if sub_idx > 0:
                output_base += f"_sub{sub_idx}"

            temp_segment = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_temp.mp4")
            cropped_vertical = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_vertical.mp4")
            final_segment = os.path.join(ctx.cfg.processing.videos_dir, f"{output_base}_final.mp4")

            os.makedirs(ctx.cfg.processing.videos_dir, exist_ok=True)

            # 3. Извлечение сегмента видео
            extract_success = self._extract_video_segment(
                ctx.video_path, temp_segment, start, end, ctx.initial_width, ctx.initial_height
            )
            if not extract_success:
                logger.logger.error(f"❌ Не удалось извлечь сегмент {start:.2f}s-{end:.2f}s")
                return None

            # 4. Создание вертикального кропа
            crop_mode = ctx.cfg.processing.crop_mode
            crop_function = crop_to_70_percent_with_blur if crop_mode == "70_percent_blur" else crop_to_vertical_average_face

            vert_crop_path = self._safe_file_operation(
                f"создание вертикального кропа для сегмента {start:.2f}s-{end:.2f}s",
                crop_function,
                temp_segment, cropped_vertical
            )
            if not vert_crop_path:
                logger.logger.error(f"❌ Не удалось создать вертикальный кроп для сегмента {start:.2f}s-{end:.2f}s")
                return None

            # 5. Добавление субтитров
            captioning_success = self._add_captions_to_short(
                cropped_vertical, final_segment,
                ctx.transcription_segments, start, end
            )

            if captioning_success:
                logger.logger.debug(f"✅ Успешно обработан сегмент {start:.2f}s-{end:.2f}s: {final_segment}")
                return final_segment
            else:
                logger.logger.error(f"❌ Не удалось добавить субтитры к сегменту {start:.2f}s-{end:.2f}s")
                return None

        except Exception as e:
            logger.logger.error(f"❌ Ошибка при обработке сегмента {start:.2f}s-{end:.2f}s: {e}")
            return None
        finally:
            # Очистка временных файлов для этого сегмента
            self._cleanup_temp_files([temp_segment, cropped_vertical], f" после обработки сегмента {sub_idx}")

    def _concatenate_segments(self, segment_paths: List[str], seq: int, ctx) -> Optional[str]:
        """Склеивание нескольких видео сегментов в один файл"""
        if not segment_paths:
            return None

        if len(segment_paths) == 1:
            # Если только один сегмент, просто возвращаем его
            return segment_paths[0]

        try:
            # Создаем финальный путь
            base_name = os.path.splitext(os.path.basename(ctx.video_path))[0]
            final_output, _ = build_short_output_name(base_name, seq, ctx.cfg.processing.shorts_dir)
            os.makedirs(ctx.cfg.processing.shorts_dir, exist_ok=True)

            # Создаем список файлов для конкатенации
            concat_list_path = os.path.join(ctx.cfg.processing.videos_dir, f"concat_list_{seq}.txt")

            with open(concat_list_path, 'w', encoding='utf-8') as f:
                for path in segment_paths:
                    # Для ffmpeg concat нужно экранировать пути
                    escaped_path = path.replace("'", "\\'")
                    f.write(f"file '{escaped_path}'\n")

            # Выполняем конкатенацию через ffmpeg
            import subprocess
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                "-y",
                final_output
            ]

            logger.logger.info(f"Склеивание {len(segment_paths)} сегментов в {final_output}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.logger.info(f"✅ Успешно склеено {len(segment_paths)} сегментов")
                # Очистка списка конкатенации
                try:
                    os.remove(concat_list_path)
                except:
                    pass
                return final_output
            else:
                logger.logger.error(f"❌ Ошибка при склеивании сегментов: {result.stderr}")
                return None

        except Exception as e:
            logger.logger.error(f"❌ Исключение при склеивании сегментов: {e}")
            return None

    def _safe_file_operation(self, operation_name: str, operation_func, *args, **kwargs) -> Optional[Any]:
        """Безопасное выполнение файловой операции с graceful degradation"""
        try:
            logger.logger.debug(f"Выполнение операции: {operation_name}")
            result = operation_func(*args, **kwargs)
            logger.logger.debug(f"✅ Операция {operation_name} выполнена успешно")
            return result
        except Exception as e:
            logger.logger.warning(f"⚠️ Операция {operation_name} завершилась с ошибкой: {e}")
            logger.logger.warning(f"Продолжаем выполнение несмотря на ошибку в {operation_name}")
            return None

    def _cleanup_temp_files(self, files_to_clean: List[str], context: str = "") -> None:
        """Безопасная очистка временных файлов"""
        if not files_to_clean:
            return

        logger.logger.info(f"Очистка временных файлов{context}...")
        for temp_file in files_to_clean:
            if temp_file and os.path.exists(temp_file):
                try:
                    file_size = os.path.getsize(temp_file) / (1024 * 1024)  # MB
                    os.remove(temp_file)
                    logger.logger.info(f"✅ Удален временный файл: {temp_file} ({file_size:.2f} MB)")
                except Exception as e:
                    logger.logger.warning(f"⚠️ Не удалось удалить временный файл {temp_file}: {e}")

    def _create_result(self, video_path: str, ranked_moments: List[RankedMoment], transcription_data: Dict[str, Any], generated_shorts: List[str] = None) -> FilmAnalysisResult:
        """Создание финального результата анализа"""
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        duration = transcription_data.get('duration', 0)

        # Формирование keep_ranges
        k = int(max(1, getattr(self.film_config, "generator_top_k", getattr(self.film_config, "target_shorts_count", 30))))
        top_n = min(k, len(ranked_moments))
        keep_ranges = []
        for rm in ranked_moments[:top_n]:
            keep_ranges.append({
                'start': rm.moment.start_time,
                'end': rm.moment.end_time,
                'type': rm.moment.moment_type,
                'score': round(rm.total_score, 2),
                'text': rm.moment.text[:200] + '...' if len(rm.moment.text) > 200 else rm.moment.text,
                'keywords': rm.moment.keywords or []
            })

        # Формирование scores
        scores = []
        for rm in ranked_moments[:top_n]:
            score_dict = {
                'moment_id': f"{rm.moment.moment_type.lower()}_{rm.rank}",
                'total': round(rm.total_score, 2),
                'keywords': rm.moment.keywords or []
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

        info = getattr(self, "_last_ranking_info", {}) or {}
        rc = getattr(self.film_config, "ranking", {}) or {}
        try:
            min_thr = rc.get("min_quality_threshold", getattr(self.film_config, "min_quality_score", 0.5))
        except Exception:
            min_thr = getattr(self.film_config, "min_quality_score", 0.5)
        soft_min = rc.get("soft_min_quality", 0.35)

        metadata = {
            'processed_at': datetime.now().isoformat(),
            'model_version': self.config.llm.model_name,
            'total_segments_analyzed': len(transcription_data.get('segments', [])),
            'video_duration': duration,
            'video_duration_sec': duration,
            'moments_found': len(ranked_moments),
            'shorts_generated': len(generated_shorts),
            'selection_strategy': info.get('selection_strategy', 'quality_threshold'),
            'ranking_system': 'keyword_matching',  # Новая система ранжирования
            'thresholds': {
                'min_quality_threshold': float(min_thr),
                'soft_min_quality': float(soft_min),
            },
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
# ========== Film Mode v2 helpers (windowed extraction, dedupe, extended scoring) ==========

    def _iter_windows(self, duration_sec: float, win_min: int, overlap_min: int):
        """Итератор окон (start, end) в секундах по длительности видео."""
        try:
            W = max(60.0, float(win_min) * 60.0)
        except Exception:
            W = 12 * 60.0
        try:
            O = max(0.0, float(overlap_min) * 60.0)
        except Exception:
            O = 3 * 60.0

        if duration_sec is None or duration_sec <= 0:
            return
        if W <= 0:
            W = min(12 * 60.0, max(60.0, duration_sec))

        step = max(1.0, W - O)
        t = 0.0
        while t < duration_sec:
            start = t
            end = min(duration_sec, t + W)
            if end - start >= 10.0:  # пропускаем слишком короткие окна
                yield (start, end)
            if end >= duration_sec:
                break
            t += step

    def _extract_film_moments_windowed(self, transcription_data: Dict[str, Any]) -> List[FilmMoment]:
        """
        Оконное извлечение кандидатов: идем по таймлайну окнами и для каждого окна просим LLM
        вернуть до K лучших моментов в рамках этого окна.
        """
        moments: List[FilmMoment] = []
        try:
            duration = float(transcription_data.get('duration', 0.0) or 0.0)
        except Exception:
            duration = 0.0

        if duration <= 0.0:
            logger.logger.warning("duration<=0 для windowed-извлечения — возвращаю пустой список")
            return []

        # Готовим segments в dict-формате для build_transcription_prompt
        segments_legacy = transcription_data.get('segments', []) or []
        segs_dict: List[Dict[str, Any]] = []
        for seg in segments_legacy:
            try:
                if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                    segs_dict.append({'text': str(seg[0]), 'start': float(seg[1]), 'end': float(seg[2])})
                elif isinstance(seg, dict):
                    st = float(seg.get('start', 0.0) or 0.0)
                    en = float(seg.get('end', 0.0) or 0.0)
                    tx = str(seg.get('text', '') or '')
                    segs_dict.append({'text': tx, 'start': st, 'end': en})
            except Exception:
                continue

        win_min = getattr(self.film_config, "window_minutes", 12)
        ov_min = getattr(self.film_config, "window_overlap_minutes", 3)
        k_per_win = int(max(1, getattr(self.film_config, "max_moments_per_window", 12)))
        # Увеличиваем лимит на окно для достижения целевых 30 шортов
        target_per_window = max(k_per_win, self.film_config.target_shorts_count // 4)  # минимум 4 окна
        k_per_win = min(target_per_window, 20)  # но не больше 20 на окно

        # Общая системная инструкция для окна (с выделением ключевых слов)
        def _build_window_system_instruction(w_start: float, w_end: float) -> str:
            return f"""
        Ты — эксперт по анализу фильмов для создания вирусных Shorts. Твоя задача — в пределах окна [{w_start:.2f}s, {w_end:.2f}s] найти лучшие моменты двух типов и вернуть ТОЛЬКО JSON-массив:

        Типы моментов:
        - COMBO (10–20 сек): 2–4 суб-сегмента одной сцены в хронологическом порядке
        - SINGLE (30–60 сек): самодостаточный момент с мини-аркой (завязка → нарастание → развязка)

        Для КАЖДОГО момента выдели 3-7 ключевых слов, которые характеризуют его суть и потенциал для вирусности.

        Формат каждого объекта:
        - moment_type: "COMBO" | "SINGLE"
        - start_time: секунды (абсолютные таймкоды фильма)
        - end_time: секунды
        - text: краткий текст/реплики момента
        - context: почему момент подходит (кратко)
        - keywords: массив строк с ключевыми словами (3-7 слов)
        Для COMBO:
        - segments: массив объектов {{"start":sec,"end":sec,"text":"..."}}

        Условия:
        - Верни до {k_per_win} лучших моментов ТОЛЬКО в границах окна.
        - Таймкоды указывай абсолютные, соответствующие фильму.
        - Строго JSON без пояснений.
        - Стремись найти максимум качественных моментов для достижения цели в 30 шортов.
        """

        # Идем по окнам
        total_windows = 0
        for w_start, w_end in self._iter_windows(duration, win_min, ov_min):
            total_windows += 1
            # Подтягиваем сегменты окна
            win_segments = []
            for s in segs_dict:
                try:
                    if s['end'] > w_start and s['start'] < w_end:
                        win_segments.append(s)
                except Exception:
                    continue

            if not win_segments:
                logger.logger.debug(f"[WINDOW] пустое окно без сегментов: {w_start:.2f}-{w_end:.2f}")
                continue

            try:
                # Текст окна
                trans_text = build_transcription_prompt(win_segments)
                system_instruction = _build_window_system_instruction(w_start, w_end)
                generation_config = make_generation_config(system_instruction, temperature=self.film_config.llm_temperature)

                response = call_llm_with_film_mode_retry(
                    system_instruction=None,
                    content=trans_text,
                    generation_config=generation_config,
                    model=self.film_config.llm_model,
                    max_api_attempts=5,
                )
                if not response or not getattr(response, "text", None):
                    logger.logger.warning(f"[WINDOW] пустой ответ LLM на окно {w_start:.2f}-{w_end:.2f}")
                    continue

                resp = response.text.strip()
                if resp.startswith("```json"):
                    resp = resp[7:].strip()
                if resp.endswith("```"):
                    resp = resp[:-3].strip()

                try:
                    arr = json.loads(resp)
                except json.JSONDecodeError as je:
                    logger.logger.warning(f"[WINDOW] JSONDecodeError в окне {w_start:.2f}-{w_end:.2f}: {je}")
                    logger.logger.debug(f"RAW: {resp[:300]}...")
                    continue

                if not isinstance(arr, list):
                    logger.logger.warning(f"[WINDOW] LLM вернул не массив в окне {w_start:.2f}-{w_end:.2f}")
                    continue

                for i, item in enumerate(arr):
                    try:
                        if not isinstance(item, dict):
                            continue
                        st = float(item.get('start_time', 0.0) or 0.0)
                        en = float(item.get('end_time', 0.0) or 0.0)
                        # Жестко клиппим в границы видео
                        st = max(0.0, min(st, duration))
                        en = max(0.0, min(en, duration))
                        if en <= st:
                            continue
                        mtype = str(item.get('moment_type', 'SINGLE') or 'SINGLE').upper()
                        txt = str(item.get('text', '') or '')
                        ctx = str(item.get('context', '') or '')
                        segs = item.get('segments', [])
                        keywords = item.get('keywords', [])
                        # Валидация по типу
                        if mtype == 'COMBO':
                            if not isinstance(segs, list) or len(segs) < getattr(self.film_config, "min_combo_segments", 2):
                                # если плохие sub-сегменты — конвертируем в SINGLE
                                mtype = 'SINGLE'
                        # Приводим sub-сегменты к ожидаемому формату (если есть)
                        norm_segments: List[Dict[str, Any]] = []
                        if isinstance(segs, list):
                            for ss in segs:
                                try:
                                    if isinstance(ss, dict):
                                        sst = float(ss.get('start', st))
                                        sse = float(ss.get('end', en))
                                        sst = max(0.0, min(sst, duration))
                                        sse = max(0.0, min(sse, duration))
                                        if sse > sst:
                                            norm_segments.append({'start': sst, 'end': sse, 'text': str(ss.get('text', '') or '')})
                                except Exception:
                                    continue

                        fm = FilmMoment(
                            moment_type=mtype,
                            start_time=st,
                            end_time=en,
                            text=txt,
                            segments=norm_segments,
                            context=ctx,
                            keywords=keywords
                        )
                        moments.append(fm)
                    except Exception:
                        continue

            except Exception as e:
                logger.logger.warning(f"[WINDOW] ошибка при обработке окна {w_start:.2f}-{w_end:.2f}: {e}")
                continue

        logger.logger.info(f"[WINDOW] собрано сырых кандидатов: {len(moments)} из {total_windows} окон")

        # Дедупликация/слияние
        deduped = self._dedupe_and_merge_moments(moments)
        logger.logger.info(f"[WINDOW] после дедупликации: {len(deduped)}")
        return deduped

    def _interval_iou(self, a0: float, a1: float, b0: float, b1: float) -> float:
        """IoU для одномерных интервалов [a0, a1] и [b0, b1]."""
        try:
            if a1 <= a0 or b1 <= b0:
                return 0.0
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            uni = max(a1, b1) - min(a0, b0)
            if uni <= 0:
                return 0.0
            return inter / uni
        except Exception:
            return 0.0

    def _dedupe_and_merge_moments(self, candidates: List[FilmMoment]) -> List[FilmMoment]:
        """
        Простая NMS-подобная дедупликация по времени с выбором лучшего кандидата.
        Критерий "лучше": больший total_score (оценим через текущую систему скоринга),
        при равенстве — более подходящая длительность (ближе к центру допустимого интервала).
        """
        if not candidates:
            return []

        thr = float(getattr(self.film_config, "dedupe_iou_threshold", 0.5) or 0.5)
        # Предварительная оценка total_score для сортировки (с учетом ключевых слов)
        scored_list: List[RankedMoment] = []
        for c in candidates:
            try:
                sc = self._calculate_moment_scores(c)
                total = sum(sc.get(k, 0.0) * self.film_config.ranking_weights.get(k, 0.0) for k in sc.keys())
                # Бонус за количество ключевых слов
                keyword_bonus = len(c.keywords or []) * 0.1
                total += keyword_bonus
                scored_list.append(RankedMoment(moment=c, scores=sc, total_score=total, rank=0))
            except Exception:
                scored_list.append(RankedMoment(moment=c, scores={}, total_score=0.0, rank=0))

        # Сортируем по total_score по убыванию, чтобы первыми оставить лучших
        scored_list.sort(key=lambda x: x.total_score, reverse=True)

        kept: List[RankedMoment] = []
        for rm in scored_list:
            ok = True
            for kept_rm in kept:
                iou = self._interval_iou(rm.moment.start_time, rm.moment.end_time, kept_rm.moment.start_time, kept_rm.moment.end_time)
                if iou >= thr:
                    ok = False
                    break
            if ok:
                kept.append(rm)

        # Возвращаем списком FilmMoment (без рангов)
        return [rm.moment for rm in kept]

    def _compute_pace_silence_scores(self, moment: FilmMoment, transcription_data: Dict[str, Any]) -> tuple[float, float]:
        """
        Возвращает (pace_score_0_10, silence_penalty_0_10):
        - pace_score: плотность слов/сек (нормирована на [0..10] при cap=4.0 слов/сек)
        - silence_penalty: доля длинных пауз в моменте -> [0..10]
        """
        try:
            st = float(moment.start_time)
            en = float(moment.end_time)
            dur = max(1e-6, en - st)
        except Exception:
            return (0.0, 0.0)

        # Плотность слов/сек по legacy segments
        segments = transcription_data.get('segments', []) or []
        words = 0
        overlapped_dur = 0.0
        for seg in segments:
            try:
                if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                    sst = float(seg[1]); sse = float(seg[2])
                    if sse > st and sst < en:
                        ov = max(0.0, min(sse, en) - max(sst, st))
                        if ov > 0:
                            txt = str(seg[0]) or ""
                            # Простейшая оценка количества слов: split по пробелам
                            wcnt = len([w for w in txt.strip().split() if w])
                            words += wcnt
                            overlapped_dur += ov
            except Exception:
                continue

        pace = 0.0
        if overlapped_dur > 0:
            pace_wps = float(words) / overlapped_dur
            # Нормируем: 0 -> 0, 4 слов/сек -> 10, cap
            pace = max(0.0, min(10.0, (pace_wps / 4.0) * 10.0))

        # Silence penalty: используем интеллектуальный детектор скучных сегментов
        try:
            boring = self._detect_boring_segments_in_moment(moment, transcription_data) or []
            boring_total = 0.0
            ai_trimmed_duration = 0.0

            for b in boring:
                try:
                    b0 = float(b.get('start', st)); b1 = float(b.get('end', st))
                    duration = max(0.0, b1 - b0)
                    boring_total += duration

                    # Отслеживаем длительность ИИ-обрезанных пауз
                    if b.get('reason', '').startswith('intelligent_'):
                        ai_trimmed_duration += duration
                except Exception:
                    continue

            frac = max(0.0, min(1.0, boring_total / dur))
            silence_penalty = min(10.0, frac * 10.0)

            # Логируем информацию об ИИ-анализе пауз для скоринга
            if boring:
                ai_fraction = ai_trimmed_duration / boring_total if boring_total > 0 else 0
                logger.logger.debug(f"Silence analysis for moment {st:.1f}s-{en:.1f}s: "
                                  f"total_boring={boring_total:.2f}s ({frac:.1%}), "
                                  f"ai_trimmed={ai_trimmed_duration:.2f}s ({ai_fraction:.1%})")
        except Exception as e:
            logger.logger.debug(f"Error in silence penalty calculation: {e}")
            silence_penalty = 0.0

        return (pace, silence_penalty)

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
            metadata={'error': True, 'ranking_system': 'keyword_matching'},
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