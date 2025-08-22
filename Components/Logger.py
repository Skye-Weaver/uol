"""
Улучшенная система логирования и мониторинга ресурсов для проекта обработки видео.
Включает детальное логирование, мониторинг GPU/CPU, прогресс-бары и ротацию логов.
"""

import time
import logging
import psutil
import GPUtil
from functools import wraps
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import json
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from dataclasses import dataclass, field
from contextlib import contextmanager
from tqdm import tqdm
import os
from logging.handlers import RotatingFileHandler
import sys
from Components.ResourceMonitor import resource_monitor, ResourceAlert


@dataclass
class PerformanceMetrics:
    """Метрики производительности для операций"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    cpu_usage_start: float = 0.0
    cpu_usage_end: float = 0.0
    memory_usage_start: int = 0
    memory_usage_end: int = 0
    gpu_usage_start: Optional[Dict[str, float]] = None
    gpu_usage_end: Optional[Dict[str, float]] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Длительность операции в секундах"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def cpu_delta(self) -> float:
        """Изменение CPU использования в процентах"""
        return self.cpu_usage_end - self.cpu_usage_start

    @property
    def memory_delta(self) -> int:
        """Изменение памяти в байтах"""
        return self.memory_usage_end - self.memory_usage_start


class ResourceMonitor:
    """Мониторинг системных ресурсов"""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0

    def get_cpu_usage(self) -> float:
        """Получить текущее использование CPU в процентах"""
        return psutil.cpu_percent(interval=0.1)

    def get_memory_usage(self) -> int:
        """Получить текущее использование памяти в байтах"""
        return psutil.Process().memory_info().rss

    def get_gpu_usage(self) -> Optional[Dict[str, float]]:
        """Получить использование GPU"""
        if not self.gpu_available:
            return None

        try:
            gpus = GPUtil.getGPUs()
            gpu_data = {}
            for i, gpu in enumerate(gpus):
                gpu_data[f'gpu_{i}'] = {
                    'usage': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                }
            return gpu_data
        except Exception:
            return None

    def get_system_info(self) -> Dict[str, Any]:
        """Получить общую информацию о системе"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'gpu_available': self.gpu_available,
            'gpu_count': self.gpu_count,
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda if self.gpu_available else None
        }


class AdvancedLogger:
    """Улучшенный логгер с мониторингом ресурсов и таймингами"""

    def __init__(self, name: str = "VideoProcessor", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Используем глобальный монитор ресурсов
        self.resource_monitor = resource_monitor
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()

        # Настройка логгеров
        self._setup_loggers()

        # Thread pool для асинхронных операций
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="Logger")

        # GPU-first настройки
        self._ensure_gpu_priority()

        # Настройка callback для оповещений о ресурсах
        self._setup_resource_alerts()

    def _setup_loggers(self):
        """Настройка системы логирования с ротацией"""
        # Основной логгер
        self.logger = logging.getLogger(f"{self.name}.main")
        self.logger.setLevel(logging.DEBUG)

        # Убираем существующие обработчики
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Файловый обработчик с ротацией
        log_file = self.log_dir / f"{self.name.lower()}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)

        # Консольный обработчик
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Форматтеры
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Отдельный логгер для производительности
        self.perf_logger = logging.getLogger(f"{self.name}.performance")
        perf_file = self.log_dir / f"{self.name.lower()}_performance.log"
        perf_handler = RotatingFileHandler(
            perf_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        perf_handler.setFormatter(file_formatter)
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)

        # Отдельный логгер для ошибок
        self.error_logger = logging.getLogger(f"{self.name}.errors")
        error_file = self.log_dir / f"{self.name.lower()}_errors.log"
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setFormatter(file_formatter)
        self.error_logger.addHandler(error_handler)
        self.error_logger.setLevel(logging.ERROR)

    def _ensure_gpu_priority(self):
        """Обеспечить GPU-first подход"""
        if self.resource_monitor.gpu_available:
            try:
                # Установить текущий GPU
                torch.cuda.set_device(0)

                # Оптимизации для GPU
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

                # Очистить кэш GPU
                torch.cuda.empty_cache()

                self.logger.info(f"GPU-first режим активирован. Используется GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                self.logger.warning(f"Не удалось настроить GPU-first режим: {e}")
        else:
            self.logger.info("GPU не доступен, используется CPU режим")

    def _setup_resource_alerts(self):
        """Настройка callback функций для оповещений о ресурсах"""
        def resource_alert_handler(alert: ResourceAlert):
            """Обработчик оповещений о ресурсах"""
            alert_msg = f"{alert.severity_icon} {alert.alert_type.upper()} {alert.severity.upper()}: {alert.message}"

            if alert.severity == 'critical':
                self.logger.critical(alert_msg)
            elif alert.severity == 'warning':
                self.logger.warning(alert_msg)
            else:
                self.logger.info(alert_msg)

        self.resource_monitor.add_alert_callback(resource_alert_handler)

    def format_duration(self, seconds: float) -> str:
        """Форматирование длительности в читаемый вид"""
        if seconds < 0.001:  # < 1ms
            return ".2f"
        elif seconds < 0.1:  # < 100ms
            return ".1f"
        elif seconds < 60:  # < 1 мин
            return ".2f"
        elif seconds < 3600:  # < 1 час
            minutes = int(seconds // 60)
            secs = seconds % 60
            return ".1f"
        else:  # >= 1 час
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return ".1f"

    def format_timestamp(self, timestamp: float = None) -> str:
        """Форматирование временной метки"""
        if timestamp is None:
            timestamp = time.time()

        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%H:%M:%S.%f")[:-3]  # ЧЧ:ММ:СС.мс

    def format_operation_time(self, start_time: float, end_time: float = None) -> str:
        """Форматирование времени операции с дополнительной информацией"""
        if end_time is None:
            end_time = time.time()

        duration = end_time - start_time
        start_dt = datetime.fromtimestamp(start_time)
        end_dt = datetime.fromtimestamp(end_time)

        # Если операция длилась менее секунды, показываем миллисекунды
        if duration < 1.0:
            return ".1f"
        # Если операция длилась менее минуты, показываем секунды
        elif duration < 60.0:
            return ".2f"
        # Если операция длилась менее часа, показываем минуты и секунды
        elif duration < 3600.0:
            minutes = int(duration // 60)
            seconds = duration % 60
            return ".1f"
        # Если операция длилась час или более, показываем часы, минуты и секунды
        else:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = duration % 60
            return ".1f"

    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Начать отслеживание операции"""
        operation_id = f"{operation_name}_{time.time()}_{threading.get_ident()}"
        start_time = time.time()

        with self._lock:
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                cpu_usage_start=self.resource_monitor.get_current_status().cpu_percent if self.resource_monitor.get_current_status() else 0.0,
                memory_usage_start=self.resource_monitor.get_current_status().memory_used_gb * (1024**3) if self.resource_monitor.get_current_status() else 0,
                gpu_usage_start=self.resource_monitor.get_current_status().gpu_data if self.resource_monitor.get_current_status() else None,
                metadata=metadata or {}
            )
            self.active_operations[operation_id] = metrics

        timestamp = self.format_timestamp(start_time)
        self.logger.info(f"🚀 [{timestamp}] Начата операция: {operation_name} (ID: {operation_id})")
        return operation_id

    def end_operation(self, operation_id: str, success: bool = True, error_message: Optional[str] = None):
        """Завершить отслеживание операции"""
        end_time = time.time()

        with self._lock:
            if operation_id not in self.active_operations:
                self.logger.warning(f"Операция {operation_id} не найдена")
                return

            metrics = self.active_operations[operation_id]
            metrics.end_time = end_time
            current_status = self.resource_monitor.get_current_status()
            if current_status:
                metrics.cpu_usage_end = current_status.cpu_percent
                metrics.memory_usage_end = current_status.memory_used_gb * (1024**3)
                metrics.gpu_usage_end = current_status.gpu_data
            else:
                metrics.cpu_usage_end = 0.0
                metrics.memory_usage_end = 0
                metrics.gpu_usage_end = None
            metrics.success = success
            metrics.error_message = error_message

        # Логирование результатов
        self._log_operation_results(metrics)

        # Удаление из активных операций
        with self._lock:
            del self.active_operations[operation_id]

    def _log_operation_results(self, metrics: PerformanceMetrics):
        """Логирование результатов операции"""
        duration = metrics.duration
        cpu_delta = metrics.cpu_delta
        memory_delta = metrics.memory_delta / 1024 / 1024  # MB

        status_icon = "✅" if metrics.success else "❌"
        status_text = "УСПЕХ" if metrics.success else "ОШИБКА"

        # Форматирование времени операции
        time_info = self.format_operation_time(metrics.start_time, metrics.end_time)

        # Информация о ресурсах
        resource_info = ".1f"
        if abs(memory_delta) > 0.1:  # Показываем только значимые изменения
            resource_info += ".1f"

        # GPU информация
        gpu_info = ""
        if metrics.gpu_usage_start and metrics.gpu_usage_end:
            for gpu_id in metrics.gpu_usage_start.keys():
                if gpu_id in metrics.gpu_usage_end:
                    start_usage = metrics.gpu_usage_start[gpu_id].get('usage', 0)
                    end_usage = metrics.gpu_usage_end[gpu_id].get('usage', 0)
                    if abs(end_usage - start_usage) > 1.0:  # Показываем только значимые изменения
                        gpu_info += ".1f"

        # Основное логирование с улучшенным форматом
        timestamp = self.format_timestamp(metrics.end_time)
        self.logger.info(
            f"🏁 [{timestamp}] Операция '{metrics.operation_name}' завершена - {time_info} - {resource_info}{gpu_info} - {status_icon} {status_text}"
        )

        # Детальное логирование производительности
        perf_data = {
            'operation': metrics.operation_name,
            'duration': duration,
            'cpu_usage_start': metrics.cpu_usage_start,
            'cpu_usage_end': metrics.cpu_usage_end,
            'cpu_delta': cpu_delta,
            'memory_start_mb': metrics.memory_usage_start / 1024 / 1024,
            'memory_end_mb': metrics.memory_usage_end / 1024 / 1024,
            'memory_delta_mb': memory_delta,
            'success': metrics.success,
            'timestamp': datetime.now().isoformat(),
            'metadata': metrics.metadata
        }

        if metrics.gpu_usage_start and metrics.gpu_usage_end:
            perf_data['gpu_usage_start'] = metrics.gpu_usage_start
            perf_data['gpu_usage_end'] = metrics.gpu_usage_end

        if not metrics.success and metrics.error_message:
            perf_data['error'] = metrics.error_message

        # Асинхронная запись в файл
        self.executor.submit(self._write_performance_log, perf_data)

        # Логирование ошибок отдельно
        if not metrics.success and metrics.error_message:
            self.error_logger.error(
                f"Ошибка в операции {metrics.operation_name}: {metrics.error_message}"
            )

    def _write_performance_log(self, perf_data: Dict[str, Any]):
        """Асинхронная запись лога производительности"""
        try:
            log_entry = json.dumps(perf_data, ensure_ascii=False, indent=2)
            self.perf_logger.info(f"PERFORMANCE_DATA: {log_entry}")
        except Exception as e:
            self.logger.error(f"Ошибка записи лога производительности: {e}")

    @contextmanager
    def operation_context(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Контекстный менеджер для операций"""
        operation_id = self.start_operation(operation_name, metadata)
        try:
            yield operation_id
            self.end_operation(operation_id, success=True)
        except Exception as e:
            self.end_operation(operation_id, success=False, error_message=str(e))
            raise

    def create_progress_bar(self, total: int, desc: str = "", unit: str = "it") -> tqdm:
        """Создать прогресс-бар с мониторингом ресурсов"""
        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] '
                      '{postfix}]'
        )

    def log_system_info(self):
        """Логирование информации о системе"""
        system_info = self.monitor.get_system_info()
        self.logger.info("=== ИНФОРМАЦИЯ О СИСТЕМЕ ===")
        for key, value in system_info.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 30)

    def get_active_operations_count(self) -> int:
        """Получить количество активных операций"""
        with self._lock:
            return len(self.active_operations)

    def cleanup(self):
        """Очистка ресурсов"""
        self.executor.shutdown(wait=True)
        # Очистка GPU памяти если возможно
        if self.resource_monitor.gpu_available:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def start_resource_monitoring(self):
        """Запуск постоянного мониторинга ресурсов"""
        if not self.resource_monitor.is_monitoring:
            self.resource_monitor.start_monitoring()
            self.logger.info("🔥 Постоянный мониторинг ресурсов запущен")

    def stop_resource_monitoring(self):
        """Остановка мониторинга ресурсов"""
        if self.resource_monitor.is_monitoring:
            self.resource_monitor.stop_monitoring()
            self.logger.info("⏹️ Мониторинг ресурсов остановлен")

    def get_resource_status(self) -> Dict[str, Any]:
        """Получение текущего статуса ресурсов"""
        return self.resource_monitor.get_resource_summary()

    def log_resource_status(self):
        """Логирование текущего статуса ресурсов"""
        status = self.get_resource_status()
        if 'error' in status:
            self.logger.warning(f"Не удалось получить статус ресурсов: {status['error']}")
            return

        current = status['current']
        averages = status['averages_5min']

        self.logger.info("📊 === СТАТУС РЕСУРСОВ ===")
        self.logger.info(f"   Время: {current['timestamp']}")
        self.logger.info(f"   CPU: {current['cpu_percent']:.1f}% (средн. 5мин: {averages['cpu_percent']:.1f}%)")
        self.logger.info(f"   Память: {current['memory_percent']:.1f}% ({current['memory_used_gb']:.1f}GB / {current['memory_total_gb']:.1f}GB)")
        self.logger.info(f"   Средн. память 5мин: {averages['memory_percent']:.1f}%")

        if current['gpu_data']:
            for gpu_id, gpu_info in current['gpu_data'].items():
                self.logger.info(f"   {gpu_id}: {gpu_info.get('usage', 0):.1f}% GPU, {gpu_info.get('memory_percent', 0):.1f}% памяти, {gpu_info.get('temperature', 0):.0f}°C")

        self.logger.info(f"   Оповещений: {status['alerts_count']}")
        self.logger.info(f"   Мониторинг активен: {self.resource_monitor.is_monitoring}")
        self.logger.info("   =======================")


# Глобальный экземпляр логгера
logger = AdvancedLogger()

# Инициализация мониторинга ресурсов при импорте модуля
def initialize_resource_monitoring():
    """Инициализация постоянного мониторинга ресурсов"""
    try:
        if not logger.resource_monitor.is_monitoring:
            logger.start_resource_monitoring()
    except Exception as e:
        print(f"⚠️ Не удалось запустить мониторинг ресурсов: {e}")

# Автоматическая инициализация при импорте (можно отключить если нужно)
# initialize_resource_monitoring()


def timed_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Декоратор для отслеживания операций"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with logger.operation_context(operation_name, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_function_call(func: Callable) -> Callable:
    """Декоратор для логирования вызовов функций"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}" if hasattr(func, '__module__') else func.__name__
        logger.logger.debug(f"Вызов функции: {func_name}")
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.logger.debug(".3f")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.logger.error(".3f")
            raise
    return wrapper