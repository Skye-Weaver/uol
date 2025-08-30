"""
Интеллектуальный анализатор пауз для обрезки видео.
Использует ИИ для классификации пауз и определения их важности.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import hashlib
import os
from datetime import datetime, timedelta

from Components.LanguageTasks import call_llm_with_retry, make_generation_config
from Components.config import get_config, IntelligentPauseAnalysisConfig
from Components.Logger import logger


@dataclass
class PauseAnalysis:
    """Результат анализа одной паузы"""
    start_time: float
    end_time: float
    duration: float
    category: str  # "structural", "filler", "emphasis", "breathing"
    confidence: float  # 0.0 to 1.0
    importance_score: float  # -1.0 to 1.0 (отрицательный = менее важный)
    should_trim: bool
    reasoning: str
    context_before: str = ""
    context_after: str = ""


@dataclass
class PauseAnalysisResult:
    """Результат анализа всех пауз в транскрипции"""
    pauses: List[PauseAnalysis] = field(default_factory=list)
    total_pauses_analyzed: int = 0
    ai_processed_pauses: int = 0
    legacy_processed_pauses: int = 0
    cache_hits: int = 0
    processing_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


class IntelligentPauseAnalyzer:
    """
    Интеллектуальный анализатор пауз с использованием ИИ.

    Основные возможности:
    - Классификация пауз по категориям (структурные, заполнители, для эффекта, дыхательные)
    - Определение важности пауз на основе контекста
    - Использование gemini-2.5-flash-lite для простых задач
    - Кеширование результатов анализа
    - Оптимизация под API ограничения
    """

    def __init__(self, config: IntelligentPauseAnalysisConfig):
        if not isinstance(config, IntelligentPauseAnalysisConfig):
            raise TypeError(f"Expected IntelligentPauseAnalysisConfig, got {type(config)}")
        self.config = config
        self.cache = {}  # Простой in-memory кеш
        self._load_cache_from_disk()

    def analyze_pauses_in_transcription(self, transcription_data: Dict[str, Any]) -> PauseAnalysisResult:
        """
        Основной метод анализа пауз в транскрипции.

        Args:
            transcription_data: Данные транскрипции с сегментами

        Returns:
            PauseAnalysisResult: Результаты анализа всех пауз
        """
        import time
        start_time = time.time()

        result = PauseAnalysisResult()

        try:
            # Извлекаем паузы из транскрипции
            pauses = self._extract_pauses_from_transcription(transcription_data)
            result.total_pauses_analyzed = len(pauses)

            if not pauses:
                logger.logger.info("Анализ пауз: паузы не найдены в транскрипции")
                return result

            logger.logger.info(f"Анализ пауз: найдено {len(pauses)} пауз для анализа")

            # Обрабатываем паузы с использованием ИИ или легаси-метода
            analyzed_pauses = []

            if self.config.enabled:
                # Используем ИИ-анализ
                analyzed_pauses = self._analyze_pauses_with_ai(pauses, transcription_data, result)
            else:
                # Используем легаси-метод
                analyzed_pauses = self._analyze_pauses_legacy(pauses, transcription_data, result)

            result.pauses = analyzed_pauses
            result.processing_time_seconds = time.time() - start_time

            # Сохраняем кеш на диск
            self._save_cache_to_disk()

            logger.logger.info(f"Анализ пауз завершен: {len(analyzed_pauses)} пауз обработано за {result.processing_time_seconds:.2f}с")
            logger.logger.info(f"ИИ-обработка: {result.ai_processed_pauses}, Легаси: {result.legacy_processed_pauses}, Кеш: {result.cache_hits}")

        except Exception as e:
            error_msg = f"Ошибка при анализе пауз: {str(e)}"
            logger.logger.error(error_msg)
            result.errors.append(error_msg)
            result.processing_time_seconds = time.time() - start_time

        return result

    def _extract_pauses_from_transcription(self, transcription_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Извлекает паузы из данных транскрипции.

        Returns:
            List[Dict]: Список пауз с start_time, end_time, duration
        """
        pauses = []
        segments = transcription_data.get('segments', [])

        if not segments:
            return pauses

        # Сортируем сегменты по времени
        sorted_segments = sorted(segments, key=lambda x: x.get('start', 0) if isinstance(x, dict) else 0)

        for i in range(len(sorted_segments) - 1):
            current_segment = sorted_segments[i]
            next_segment = sorted_segments[i + 1]

            # Извлекаем времена окончания и начала
            if isinstance(current_segment, dict) and isinstance(next_segment, dict):
                current_end = current_segment.get('end', 0)
                next_start = next_segment.get('start', 0)

                if next_start > current_end:
                    duration = next_start - current_end
                    pauses.append({
                        'start_time': current_end,
                        'end_time': next_start,
                        'duration': duration,
                        'context_before': current_segment.get('text', ''),
                        'context_after': next_segment.get('text', '')
                    })

        # Фильтруем паузы по минимальной длительности
        min_pause_duration = 0.1  # 100ms минимум
        pauses = [p for p in pauses if p['duration'] >= min_pause_duration]

        return pauses

    def _analyze_pauses_with_ai(self, pauses: List[Dict[str, Any]], transcription_data: Dict[str, Any],
                               result: PauseAnalysisResult) -> List[PauseAnalysis]:
        """
        Анализирует паузы с использованием ИИ.
        """
        analyzed_pauses = []

        # Группируем паузы в батчи для оптимизации API
        batches = self._create_pause_batches(pauses)

        for batch in batches:
            try:
                batch_results = self._analyze_pause_batch(batch, transcription_data, result)
                analyzed_pauses.extend(batch_results)

                # Проверяем лимиты API
                if self._should_apply_rate_limit():
                    import time
                    delay = self.config.api_optimization.get('rate_limit_delay', 1.0)
                    logger.logger.debug(f"Применяем задержку API: {delay}с")
                    time.sleep(delay)

            except Exception as e:
                logger.logger.warning(f"Ошибка при анализе батча пауз: {e}")
                # Откатываемся на легаси-метод для этого батча
                if self.config.api_optimization.get('fallback_to_legacy', True):
                    logger.logger.info("Откатываемся на легаси-метод для батча")
                    legacy_results = self._analyze_pauses_legacy(batch, transcription_data, result)
                    analyzed_pauses.extend(legacy_results)

        return analyzed_pauses

    def _create_pause_batches(self, pauses: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Группирует паузы в батчи для оптимизации API.
        """
        batch_size = self.config.batch_size
        return [pauses[i:i + batch_size] for i in range(0, len(pauses), batch_size)]

    def _analyze_pause_batch(self, batch: List[Dict[str, Any]], transcription_data: Dict[str, Any],
                           result: PauseAnalysisResult) -> List[PauseAnalysis]:
        """
        Анализирует батч пауз с использованием ИИ.
        """
        # Проверяем кеш
        cache_key = self._generate_cache_key(batch, transcription_data)
        if self.config.cache_enabled and cache_key in self.cache:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                result.cache_hits += len(cached_result)
                logger.logger.debug(f"Кеш hit для батча из {len(cached_result)} пауз")
                return cached_result

        # Создаем промпт для ИИ
        prompt = self._create_pause_analysis_prompt(batch, transcription_data)

        try:
            # Используем gemini-2.5-flash-lite для простых задач
            generation_config = make_generation_config(
                system_instruction=self._get_pause_analysis_system_prompt(),
                temperature=self.config.temperature
            )

            response = call_llm_with_retry(
                system_instruction=None,
                content=prompt,
                generation_config=generation_config,
                model=self.config.model,
                max_api_attempts=self.config.max_attempts
            )

            if not response or not response.text:
                raise ValueError("Пустой ответ от ИИ")

            # Парсим ответ
            analysis_results = self._parse_pause_analysis_response(response.text)

            # Создаем объекты PauseAnalysis
            pause_analyses = []
            for i, analysis_data in enumerate(analysis_results):
                if i < len(batch):
                    pause_data = batch[i]
                    analysis = self._create_pause_analysis_from_data(pause_data, analysis_data)
                    pause_analyses.append(analysis)

            result.ai_processed_pauses += len(pause_analyses)

            # Кешируем результат
            if self.config.cache_enabled:
                self._cache_result(cache_key, pause_analyses)

            return pause_analyses

        except Exception as e:
            logger.logger.warning(f"Ошибка при ИИ-анализе пауз: {e}")
            # Откатываемся на легаси-метод
            if self.config.api_optimization.get('fallback_to_legacy', True):
                return self._analyze_pauses_legacy(batch, transcription_data, result)
            else:
                # Возвращаем пустые результаты
                return []

    def _create_pause_analysis_prompt(self, batch: List[Dict[str, Any]], transcription_data: Dict[str, Any]) -> str:
        """
        Создает промпт для анализа пауз.
        """
        prompt_parts = []

        # Добавляем контекст транскрипции
        prompt_parts.append("КОНТЕКСТ ТРАНСКРИПЦИИ:")
        segments = transcription_data.get('segments', [])[:50]  # Ограничиваем для экономии токенов
        for seg in segments:
            if isinstance(seg, dict):
                start = seg.get('start', 0)
                text = seg.get('text', '')
                prompt_parts.append(".2f")

        prompt_parts.append("\nПАУЗЫ ДЛЯ АНАЛИЗА:")
        for i, pause in enumerate(batch):
            prompt_parts.append(f"Пауза {i+1}:")
            prompt_parts.append(f"  Время: {pause['start_time']:.2f}s - {pause['end_time']:.2f}s")
            prompt_parts.append(f"  Длительность: {pause['duration']:.2f}s")
            prompt_parts.append(f"  Контекст до: {pause.get('context_before', '')[:100]}...")
            prompt_parts.append(f"  Контекст после: {pause.get('context_after', '')[:100]}...")
            prompt_parts.append("")

        return "\n".join(prompt_parts)

    def _get_pause_analysis_system_prompt(self) -> str:
        """
        Возвращает системный промпт для анализа пауз.
        """
        categories = self.config.pause_categories

        return f"""
Ты — эксперт по анализу речевых пауз в транскрипциях. Твоя задача — классифицировать паузы и определить, следует ли их обрезать.

КАТЕГОРИИ ПАУЗ:
- structural: Структурные паузы (конец предложения, смена темы) — НЕ ОБРЕЗАТЬ
- filler: Заполнители речи ("э-э", "м-м", паузы размышления) — ОБРЕЗАТЬ
- emphasis: Паузы для эффекта (драматические паузы) — АНАЛИЗИРОВАТЬ КОНТЕКСТ
- breathing: Дыхательные паузы — ОБРЕЗАТЬ при длительности > 1.5с

КРИТЕРИИ ОЦЕНКИ:
1. Длительность паузы
2. Контекст до и после паузы
3. Лингвистические маркеры
4. Цель паузы (структурная vs случайная)

ВЕРНИ ТОЛЬКО JSON-массив объектов с полями:
- category: одна из категорий выше
- confidence: уверенность от 0.0 до 1.0
- importance_score: от -1.0 (не важен) до 1.0 (очень важен)
- should_trim: true/false
- reasoning: краткое объяснение решения

Пример ответа:
[
  {{
    "category": "filler",
    "confidence": 0.9,
    "importance_score": -0.7,
    "should_trim": true,
    "reasoning": "Короткая пауза размышления между словами"
  }}
]
"""

    def _parse_pause_analysis_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Парсит ответ ИИ с анализом пауз.
        """
        try:
            # Извлекаем JSON из ответа
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Ищем JSON-массив в тексте
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                if start != -1 and end > start:
                    json_str = response_text[start:end]
                else:
                    json_str = response_text.strip()

            data = json.loads(json_str)
            if isinstance(data, list):
                return data
            else:
                logger.logger.warning("ИИ вернул не массив в ответе анализа пауз")
                return []

        except Exception as e:
            logger.logger.warning(f"Ошибка парсинга ответа ИИ: {e}")
            return []

    def _create_pause_analysis_from_data(self, pause_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> PauseAnalysis:
        """
        Создает объект PauseAnalysis из данных паузы и анализа.
        """
        return PauseAnalysis(
            start_time=pause_data['start_time'],
            end_time=pause_data['end_time'],
            duration=pause_data['duration'],
            category=analysis_data.get('category', 'unknown'),
            confidence=float(analysis_data.get('confidence', 0.5)),
            importance_score=float(analysis_data.get('importance_score', 0.0)),
            should_trim=bool(analysis_data.get('should_trim', False)),
            reasoning=analysis_data.get('reasoning', ''),
            context_before=pause_data.get('context_before', ''),
            context_after=pause_data.get('context_after', '')
        )

    def _analyze_pauses_legacy(self, pauses: List[Dict[str, Any]], transcription_data: Dict[str, Any],
                              result: PauseAnalysisResult) -> List[PauseAnalysis]:
        """
        Легаси-метод анализа пауз без ИИ.
        """
        analyzed_pauses = []

        for pause_data in pauses:
            duration = pause_data['duration']

            # Простая логика определения типа паузы
            if duration < 0.5:
                category = "filler"
                should_trim = True
                importance_score = -0.5
                reasoning = "Короткая пауза, вероятно заполнитель"
            elif duration > 2.0:
                category = "structural"
                should_trim = False
                importance_score = 0.8
                reasoning = "Длинная пауза, вероятно структурная"
            else:
                category = "breathing"
                should_trim = duration > 1.5
                importance_score = 0.0
                reasoning = "Средняя пауза, возможно дыхательная"

            analysis = PauseAnalysis(
                start_time=pause_data['start_time'],
                end_time=pause_data['end_time'],
                duration=duration,
                category=category,
                confidence=0.7,  # Фиксированная уверенность для легаси
                importance_score=importance_score,
                should_trim=should_trim,
                reasoning=reasoning,
                context_before=pause_data.get('context_before', ''),
                context_after=pause_data.get('context_after', '')
            )

            analyzed_pauses.append(analysis)

        result.legacy_processed_pauses += len(analyzed_pauses)
        return analyzed_pauses

    def _generate_cache_key(self, batch: List[Dict[str, Any]], transcription_data: Dict[str, Any]) -> str:
        """
        Генерирует ключ кеша для батча пауз.
        """
        # Создаем хеш на основе контекста пауз
        cache_data = {
            'pause_times': [(p['start_time'], p['end_time']) for p in batch],
            'transcription_hash': self._get_transcription_hash(transcription_data)
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_transcription_hash(self, transcription_data: Dict[str, Any]) -> str:
        """
        Генерирует хеш транскрипции для кеширования.
        """
        segments = transcription_data.get('segments', [])
        if not segments:
            return "empty"

        # Берем первые и последние несколько сегментов для хеша
        sample_segments = segments[:3] + segments[-3:] if len(segments) > 6 else segments
        sample_text = json.dumps(sample_segments, sort_keys=True)
        return hashlib.md5(sample_text.encode()).hexdigest()

    def _cache_result(self, key: str, results: List[PauseAnalysis]) -> None:
        """
        Кеширует результаты анализа.
        """
        if not self.config.cache_enabled:
            return

        cache_entry = {
            'results': [
                {
                    'start_time': p.start_time,
                    'end_time': p.end_time,
                    'duration': p.duration,
                    'category': p.category,
                    'confidence': p.confidence,
                    'importance_score': p.importance_score,
                    'should_trim': p.should_trim,
                    'reasoning': p.reasoning,
                    'context_before': p.context_before,
                    'context_after': p.context_after
                } for p in results
            ],
            'timestamp': datetime.now().isoformat(),
            'ttl_hours': self.config.cache_ttl_hours
        }

        self.cache[key] = cache_entry

    def _get_cached_result(self, key: str) -> Optional[List[PauseAnalysis]]:
        """
        Получает результат из кеша.
        """
        if key not in self.cache:
            return None

        cache_entry = self.cache[key]

        # Проверяем срок действия кеша
        timestamp = datetime.fromisoformat(cache_entry['timestamp'])
        ttl_hours = cache_entry.get('ttl_hours', 24)
        if datetime.now() - timestamp > timedelta(hours=ttl_hours):
            # Кеш устарел
            del self.cache[key]
            return None

        # Восстанавливаем объекты PauseAnalysis
        results = []
        for data in cache_entry['results']:
            analysis = PauseAnalysis(
                start_time=data['start_time'],
                end_time=data['end_time'],
                duration=data['duration'],
                category=data['category'],
                confidence=data['confidence'],
                importance_score=data['importance_score'],
                should_trim=data['should_trim'],
                reasoning=data['reasoning'],
                context_before=data.get('context_before', ''),
                context_after=data.get('context_after', '')
            )
            results.append(analysis)

        return results

    def _load_cache_from_disk(self) -> None:
        """
        Загружает кеш с диска.
        """
        try:
            cache_file = os.path.join(os.getcwd(), "pause_analysis_cache.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.logger.debug(f"Загружен кеш анализа пауз: {len(self.cache)} записей")
        except Exception as e:
            logger.logger.warning(f"Не удалось загрузить кеш анализа пауз: {e}")
            self.cache = {}

    def _save_cache_to_disk(self) -> None:
        """
        Сохраняет кеш на диск.
        """
        try:
            cache_file = os.path.join(os.getcwd(), "pause_analysis_cache.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.logger.debug(f"Сохранен кеш анализа пауз: {len(self.cache)} записей")
        except Exception as e:
            logger.logger.warning(f"Не удалось сохранить кеш анализа пауз: {e}")

    def _should_apply_rate_limit(self) -> bool:
        """
        Определяет, нужно ли применять ограничение частоты запросов.
        """
        # Простая логика: применяем задержку после каждого батча
        return self.config.api_optimization.get('use_batch_processing', True)


# Вспомогательные функции
def create_intelligent_pause_analyzer() -> IntelligentPauseAnalyzer:
    """
    Создает экземпляр IntelligentPauseAnalyzer с настройками из конфигурации.
    """
    config = get_config()
    if not hasattr(config, 'film_mode') or config.film_mode is None:
        raise ValueError("Конфигурация film_mode отсутствует")
    if not hasattr(config.film_mode, 'intelligent_pause_analysis') or config.film_mode.intelligent_pause_analysis is None:
        raise ValueError("Конфигурация intelligent_pause_analysis отсутствует")
    return IntelligentPauseAnalyzer(config.film_mode.intelligent_pause_analysis)


def analyze_pauses_in_moment(moment_start: float, moment_end: float, transcription_data: Dict[str, Any]) -> List[PauseAnalysis]:
    """
    Анализирует паузы в рамках конкретного момента видео.

    Args:
        moment_start: Начало момента (секунды)
        moment_end: Конец момента (секунды)
        transcription_data: Данные транскрипции

    Returns:
        List[PauseAnalysis]: Анализ пауз в рамках момента
    """
    analyzer = create_intelligent_pause_analyzer()
    result = analyzer.analyze_pauses_in_transcription(transcription_data)

    # Фильтруем паузы, которые пересекаются с моментом
    moment_pauses = []
    for pause in result.pauses:
        if pause.end_time > moment_start and pause.start_time < moment_end:
            # Ограничиваем паузу границами момента
            effective_start = max(pause.start_time, moment_start)
            effective_end = min(pause.end_time, moment_end)

            adjusted_pause = PauseAnalysis(
                start_time=effective_start,
                end_time=effective_end,
                duration=effective_end - effective_start,
                category=pause.category,
                confidence=pause.confidence,
                importance_score=pause.importance_score,
                should_trim=pause.should_trim,
                reasoning=pause.reasoning,
                context_before=pause.context_before,
                context_after=pause.context_after
            )
            moment_pauses.append(adjusted_pause)

    return moment_pauses
