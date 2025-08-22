#!/usr/bin/env python3
"""
Тестовый скрипт для проверки системы логирования и мониторинга ресурсов.
"""

import time
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from Components.Logger import logger, timed_operation
from Components.config import get_config

def test_basic_logging():
    """Тест базового функционала логирования"""
    print("=== Тест базового логирования ===")

    logger.logger.info("Тестовое информационное сообщение")
    logger.logger.warning("Тестовое предупреждение")
    logger.logger.error("Тестовая ошибка")
    logger.logger.debug("Тестовое отладочное сообщение")

    print("✓ Базовое логирование работает")

def test_operation_timing():
    """Тест тайминга операций"""
    print("\n=== Тест тайминга операций ===")

    @timed_operation("test_operation")
    def dummy_operation():
        time.sleep(0.5)
        return "результат"

    result = dummy_operation()
    print(f"✓ Операция выполнена с результатом: {result}")

def test_resource_monitoring():
    """Тест мониторинга ресурсов"""
    print("\n=== Тест мониторинга ресурсов ===")

    with logger.operation_context("resource_test", {"test_type": "resource_monitoring"}):
        # Выполняем некоторую работу для генерации нагрузки
        data = [i ** 2 for i in range(10000)]
        time.sleep(0.2)

    print("✓ Мониторинг ресурсов работает")

def test_progress_bar():
    """Тест прогресс-бара"""
    print("\n=== Тест прогресс-бара ===")

    progress_bar = logger.create_progress_bar(total=10, desc="Тест прогресса", unit="итерация")

    for i in range(10):
        time.sleep(0.1)
        progress_bar.update(1)
        progress_bar.set_postfix({"текущий": i+1, "всего": 10})

    progress_bar.close()
    print("✓ Прогресс-бар работает")

def test_system_info():
    """Тест логирования системной информации"""
    print("\n=== Тест системной информации ===")

    logger.log_system_info()
    print("✓ Системная информация залогирована")

def test_gpu_monitoring():
    """Тест мониторинга GPU"""
    print("\n=== Тест мониторинга GPU ===")

    gpu_info = logger.monitor.get_gpu_usage()
    if gpu_info:
        logger.logger.info(f"GPU информация: {gpu_info}")
        print("✓ GPU мониторинг работает")
    else:
        logger.logger.info("GPU не доступен или не найден")
        print("✓ GPU мониторинг: GPU не доступен (нормально)")

def main():
    """Основная функция тестирования"""
    print("🚀 Запуск тестирования системы логирования и мониторинга\n")

    try:
        # Загружаем конфигурацию
        cfg = get_config()
        print(f"Конфигурация загружена: {cfg.logging.enable_performance_monitoring}")

        # Запускаем тесты
        test_basic_logging()
        test_operation_timing()
        test_resource_monitoring()
        test_progress_bar()
        test_system_info()
        test_gpu_monitoring()

        print("\n✅ Все тесты пройдены успешно!")
        print("📁 Проверьте папку 'logs' для просмотра лог-файлов")

        # Показываем созданные файлы
        log_dir = Path("logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            print(f"\n📄 Созданные лог-файлы ({len(log_files)}):")
            for log_file in log_files:
                size = log_file.stat().st_size
                print(f"  - {log_file.name} ({size} bytes)")

    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Очистка ресурсов
        logger.cleanup()

    return 0

if __name__ == "__main__":
    exit(main())