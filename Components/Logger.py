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
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from dataclasses import dataclass, field
from contextlib import contextmanager
from tqdm import tqdm
import os
from logging.handlers import RotatingFileHandler
import sys


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

        self.monitor = ResourceMonitor()
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()

        # Настройка логгеров
        self._setup_loggers()

        # Thread pool для асинхронных операций
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="Logger")

        # GPU-first настройки
        self._ensure_gpu_priority()

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
        if self.monitor.gpu_available:
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

    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Начать отслеживание операции"""
        operation_id = f"{operation_name}_{time.time()}_{threading.get_ident()}"

        with self._lock:
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=time.time(),
                cpu_usage_start=self.monitor.get_cpu_usage(),
                memory_usage_start=self.monitor.get_memory_usage(),
                gpu_usage_start=self.monitor.get_gpu_usage(),
                metadata=metadata or {}
            )
            self.active_operations[operation_id] = metrics

        self.logger.info(f"🚀 Начата операция: {operation_name} (ID: {operation_id})")
        return operation_id

    def end_operation(self, operation_id: str, success: bool = True, error_message: Optional[str] = None):
        """Завершить отслеживание операции"""
        with self._lock:
            if operation_id not in self.active_operations:
                self.logger.warning(f"Операция {operation_id} не найдена")
                return

            metrics = self.active_operations[operation_id]
            metrics.end_time = time.time()
            metrics.cpu_usage_end = self.monitor.get_cpu_usage()
            metrics.memory_usage_end = self.monitor.get_memory_usage()
            metrics.gpu_usage_end = self.monitor.get_gpu_usage()
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

        # Основное логирование
        self.logger.info(
            ".2f"
            ".1f"
            ".1f"
            f"{status_icon} {status_text}"
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
        if self.monitor.gpu_available:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# Глобальный экземпляр логгера
logger = AdvancedLogger()


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