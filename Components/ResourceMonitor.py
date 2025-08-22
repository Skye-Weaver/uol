"""
Модуль постоянного мониторинга ресурсов в реальном времени.
Обеспечивает непрерывное отслеживание системных ресурсов с возможностью
настройки интервалов, порогов и оповещений.
"""

import time
import threading
import psutil
import GPUtil
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import torch
from collections import deque
import json
from pathlib import Path
import logging


@dataclass
class ResourceSnapshot:
    """Снимок состояния ресурсов в момент времени"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float

    # GPU данные (если доступны)
    gpu_count: int = 0
    gpu_data: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Дополнительные метрики
    disk_usage: Optional[Dict[str, float]] = None
    network_io: Optional[Dict[str, float]] = None

    @property
    def timestamp_str(self) -> str:
        """Форматированная временная метка"""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%H:%M:%S.%f")[:-3]  # мм:сс.мс

    @property
    def is_critical(self) -> bool:
        """Проверка критических значений ресурсов"""
        return (
            self.cpu_percent > 90.0 or
            self.memory_percent > 90.0 or
            any(gpu.get('usage', 0) > 95.0 for gpu in self.gpu_data.values())
        )


@dataclass
class ResourceAlert:
    """Оповещение о состоянии ресурсов"""
    alert_type: str  # 'cpu', 'memory', 'gpu', 'disk', 'network'
    severity: str    # 'info', 'warning', 'critical'
    message: str
    value: float
    threshold: float
    timestamp: float

    @property
    def severity_icon(self) -> str:
        """Иконка для уровня серьезности"""
        icons = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'critical': '🚨'
        }
        return icons.get(self.severity, '❓')


class ResourceMonitor:
    """Монитор ресурсов с постоянным отслеживанием"""

    def __init__(self,
                 interval: float = 1.0,
                 history_size: int = 3600,  # 1 час при интервале 1 сек
                 alert_callbacks: Optional[List[Callable]] = None):
        self.interval = interval
        self.history_size = history_size
        self.alert_callbacks = alert_callbacks or []

        # История мониторинга
        self.history: deque[ResourceSnapshot] = deque(maxlen=history_size)
        self.alerts: deque[ResourceAlert] = deque(maxlen=1000)

        # Пороги оповещений
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'gpu_memory_warning': 90.0,
            'gpu_memory_critical': 95.0,
            'gpu_temp_warning': 75.0,
            'gpu_temp_critical': 85.0,
            'disk_warning': 90.0,
            'disk_critical': 95.0
        }

        # Состояние мониторинга
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # GPU информация
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0

        # Статистика
        self.stats = {
            'snapshots_taken': 0,
            'alerts_generated': 0,
            'monitoring_start_time': None,
            'last_snapshot_time': None
        }

    def start_monitoring(self):
        """Запуск постоянного мониторинга"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.stop_event.clear()
        self.stats['monitoring_start_time'] = time.time()

        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ResourceMonitor",
            daemon=True
        )
        self.monitor_thread.start()

        print("🔥 Запущен постоянный мониторинг ресурсов")
        print(f"   Интервал: {self.interval} сек")
        print(f"   GPU доступен: {'Да' if self.gpu_available else 'Нет'}")
        if self.gpu_available:
            print(f"   GPU устройств: {self.gpu_count}")

    def stop_monitoring(self):
        """Остановка мониторинга"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self.stop_event.set()

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

        print("⏹️  Остановлен мониторинг ресурсов")
        print(f"   Всего снимков: {self.stats['snapshots_taken']}")
        print(f"   Оповещений: {self.stats['alerts_generated']}")

    def _monitoring_loop(self):
        """Основной цикл мониторинга"""
        while not self.stop_event.is_set():
            try:
                snapshot = self._take_snapshot()
                self.history.append(snapshot)
                self.stats['snapshots_taken'] += 1
                self.stats['last_snapshot_time'] = snapshot.timestamp

                # Проверка порогов и генерация оповещений
                self._check_thresholds(snapshot)

                # Ожидание до следующего интервала
                if self.stop_event.wait(timeout=self.interval):
                    break

            except Exception as e:
                print(f"❌ Ошибка в цикле мониторинга: {e}")
                time.sleep(1.0)

    def _take_snapshot(self) -> ResourceSnapshot:
        """Создание снимка текущего состояния ресурсов"""
        timestamp = time.time()

        # CPU и память
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)

        snapshot = ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            gpu_count=self.gpu_count
        )

        # GPU данные
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    snapshot.gpu_data[f'gpu_{i}'] = {
                        'usage': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                        'temperature': gpu.temperature
                    }
            except Exception as e:
                print(f"⚠️  Ошибка получения GPU данных: {e}")

        # Дисковое пространство
        try:
            disk = psutil.disk_usage('/')
            snapshot.disk_usage = {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent': disk.percent
            }
        except Exception:
            pass

        # Сетевая активность
        try:
            net_io = psutil.net_io_counters()
            snapshot.network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        except Exception:
            pass

        return snapshot

    def _check_thresholds(self, snapshot: ResourceSnapshot):
        """Проверка порогов и генерация оповещений"""
        alerts = []

        # CPU проверки
        if snapshot.cpu_percent >= self.thresholds['cpu_critical']:
            alerts.append(ResourceAlert(
                'cpu', 'critical',
                f"Критическое использование CPU: {snapshot.cpu_percent:.1f}% (порог: {self.thresholds['cpu_critical']}%)",
                snapshot.cpu_percent, self.thresholds['cpu_critical'],
                snapshot.timestamp
            ))
        elif snapshot.cpu_percent >= self.thresholds['cpu_warning']:
            alerts.append(ResourceAlert(
                'cpu', 'warning',
                f"Высокое использование CPU: {snapshot.cpu_percent:.1f}% (порог: {self.thresholds['cpu_warning']}%)",
                snapshot.cpu_percent, self.thresholds['cpu_warning'],
                snapshot.timestamp
            ))

        # Память проверки
        if snapshot.memory_percent >= self.thresholds['memory_critical']:
            alerts.append(ResourceAlert(
                'memory', 'critical',
                f"Критическое использование памяти: {snapshot.memory_percent:.1f}% (порог: {self.thresholds['memory_critical']}%)",
                snapshot.memory_percent, self.thresholds['memory_critical'],
                snapshot.timestamp
            ))
        elif snapshot.memory_percent >= self.thresholds['memory_warning']:
            alerts.append(ResourceAlert(
                'memory', 'warning',
                f"Высокое использование памяти: {snapshot.memory_percent:.1f}% (порог: {self.thresholds['memory_warning']}%)",
                snapshot.memory_percent, self.thresholds['memory_warning'],
                snapshot.timestamp
            ))

        # GPU проверки
        for gpu_id, gpu_data in snapshot.gpu_data.items():
            gpu_name = f"GPU {gpu_id}"

            # Использование памяти
            mem_percent = gpu_data.get('memory_percent', 0)
            if mem_percent >= self.thresholds['gpu_memory_critical']:
                alerts.append(ResourceAlert(
                    'gpu', 'critical',
                    f"Критическое использование GPU памяти {gpu_name}: {mem_percent:.1f}% (порог: {self.thresholds['gpu_memory_critical']}%)",
                    mem_percent, self.thresholds['gpu_memory_critical'],
                    snapshot.timestamp
                ))
            elif mem_percent >= self.thresholds['gpu_memory_warning']:
                alerts.append(ResourceAlert(
                    'gpu', 'warning',
                    f"Высокое использование GPU памяти {gpu_name}: {mem_percent:.1f}% (порог: {self.thresholds['gpu_memory_warning']}%)",
                    mem_percent, self.thresholds['gpu_memory_warning'],
                    snapshot.timestamp
                ))

            # Температура
            temp = gpu_data.get('temperature', 0)
            if temp >= self.thresholds['gpu_temp_critical']:
                alerts.append(ResourceAlert(
                    'gpu', 'critical',
                    f"Критическая температура GPU {gpu_name}: {temp:.0f}°C (порог: {self.thresholds['gpu_temp_critical']}°C)",
                    temp, self.thresholds['gpu_temp_critical'],
                    snapshot.timestamp
                ))
            elif temp >= self.thresholds['gpu_temp_warning']:
                alerts.append(ResourceAlert(
                    'gpu', 'warning',
                    f"Высокая температура GPU {gpu_name}: {temp:.0f}°C (порог: {self.thresholds['gpu_temp_warning']}°C)",
                    temp, self.thresholds['gpu_temp_warning'],
                    snapshot.timestamp
                ))

        # Диск проверки
        if snapshot.disk_usage and snapshot.disk_usage['percent'] >= self.thresholds['disk_critical']:
            alerts.append(ResourceAlert(
                'disk', 'critical',
                f"Критическое использование диска: {snapshot.disk_usage['percent']:.1f}% (порог: {self.thresholds['disk_critical']}%)",
                snapshot.disk_usage['percent'], self.thresholds['disk_critical'],
                snapshot.timestamp
            ))
        elif snapshot.disk_usage and snapshot.disk_usage['percent'] >= self.thresholds['disk_warning']:
            alerts.append(ResourceAlert(
                'disk', 'warning',
                f"Высокое использование диска: {snapshot.disk_usage['percent']:.1f}% (порог: {self.thresholds['disk_warning']}%)",
                snapshot.disk_usage['percent'], self.thresholds['disk_warning'],
                snapshot.timestamp
            ))

        # Обработка оповещений
        for alert in alerts:
            self.alerts.append(alert)
            self.stats['alerts_generated'] += 1

            # Вызов callback функций
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"❌ Ошибка в callback функции: {e}")

    def get_current_status(self) -> Optional[ResourceSnapshot]:
        """Получение текущего состояния ресурсов"""
        return self.history[-1] if self.history else None

    def get_recent_history(self, seconds: int = 60) -> List[ResourceSnapshot]:
        """Получение истории за последние N секунд"""
        if not self.history:
            return []

        cutoff_time = time.time() - seconds
        return [snap for snap in self.history if snap.timestamp >= cutoff_time]

    def get_resource_summary(self) -> Dict[str, Any]:
        """Получение сводки по ресурсам"""
        if not self.history:
            return {"error": "Нет данных мониторинга"}

        current = self.history[-1]
        recent = self.get_recent_history(300)  # Последние 5 минут

        # Расчет средних значений
        avg_cpu = sum(s.cpu_percent for s in recent) / len(recent) if recent else 0
        avg_memory = sum(s.memory_percent for s in recent) / len(recent) if recent else 0

        # GPU статистика
        gpu_stats = {}
        if current.gpu_data:
            for gpu_id, gpu_data in current.gpu_data.items():
                recent_gpu = [s.gpu_data.get(gpu_id, {}) for s in recent if gpu_id in s.gpu_data]
                if recent_gpu:
                    avg_usage = sum(g.get('usage', 0) for g in recent_gpu) / len(recent_gpu)
                    avg_temp = sum(g.get('temperature', 0) for g in recent_gpu) / len(recent_gpu)
                    gpu_stats[gpu_id] = {
                        'current_usage': gpu_data.get('usage', 0),
                        'current_temp': gpu_data.get('temperature', 0),
                        'avg_usage_5min': avg_usage,
                        'avg_temp_5min': avg_temp
                    }

        return {
            'current': {
                'timestamp': current.timestamp_str,
                'cpu_percent': current.cpu_percent,
                'memory_percent': current.memory_percent,
                'memory_used_gb': current.memory_used_gb,
                'memory_total_gb': current.memory_total_gb,
                'gpu_data': current.gpu_data
            },
            'averages_5min': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory
            },
            'gpu_stats': gpu_stats,
            'alerts_count': len(self.alerts),
            'monitoring_duration': time.time() - (self.stats['monitoring_start_time'] or time.time()),
            'snapshots_count': len(self.history)
        }

    def set_threshold(self, resource: str, level: str, value: float):
        """Установка порога оповещения"""
        key = f"{resource}_{level}"
        if key in self.thresholds:
            self.thresholds[key] = value
            print(f"✅ Установлен порог {key}: {value}")
        else:
            print(f"❌ Неизвестный порог: {key}")

    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]):
        """Добавление callback функции для оповещений"""
        self.alert_callbacks.append(callback)

    def clear_history(self):
        """Очистка истории мониторинга"""
        self.history.clear()
        self.alerts.clear()
        print("🧹 История мониторинга очищена")

    def export_history(self, filepath: str, format: str = 'json'):
        """Экспорт истории мониторинга"""
        if format.lower() == 'json':
            data = {
                'snapshots': [self._snapshot_to_dict(s) for s in self.history],
                'alerts': [self._alert_to_dict(a) for a in self.alerts],
                'stats': self.stats,
                'export_time': time.time()
            }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"📊 История экспортирована в {filepath}")

    def _snapshot_to_dict(self, snapshot: ResourceSnapshot) -> Dict[str, Any]:
        """Преобразование снимка в словарь для экспорта"""
        return {
            'timestamp': snapshot.timestamp,
            'timestamp_str': snapshot.timestamp_str,
            'cpu_percent': snapshot.cpu_percent,
            'memory_percent': snapshot.memory_percent,
            'memory_used_gb': snapshot.memory_used_gb,
            'memory_total_gb': snapshot.memory_total_gb,
            'gpu_count': snapshot.gpu_count,
            'gpu_data': snapshot.gpu_data,
            'disk_usage': snapshot.disk_usage,
            'network_io': snapshot.network_io
        }

    def _alert_to_dict(self, alert: ResourceAlert) -> Dict[str, Any]:
        """Преобразование оповещения в словарь для экспорта"""
        return {
            'alert_type': alert.alert_type,
            'severity': alert.severity,
            'message': alert.message,
            'value': alert.value,
            'threshold': alert.threshold,
            'timestamp': alert.timestamp,
            'timestamp_str': datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        }


# Глобальный монитор ресурсов
resource_monitor = ResourceMonitor()