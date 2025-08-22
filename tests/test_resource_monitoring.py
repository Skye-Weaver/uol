#!/usr/bin/env python3
"""
Тест новой системы мониторинга ресурсов в реальном времени.
Проверяет работу ResourceMonitor и интеграцию с AdvancedLogger.
"""

import time
import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь для импортов
sys.path.insert(0, str(Path(__file__).parent))

from Components.Logger import logger, timed_operation
from Components.ResourceMonitor import resource_monitor


def test_basic_monitoring():
    """Тест базового функционала мониторинга"""
    print("🧪 Тестирование базового мониторинга ресурсов...")

    # Запуск мониторинга
    logger.start_resource_monitoring()

    # Ждем несколько секунд для накопления данных
    print("   Ожидание 5 секунд для сбора данных...")
    time.sleep(5)

    # Получение статуса
    status = logger.get_resource_status()
    print(f"   Статус получен: {status.get('current', {}).get('timestamp', 'N/A')}")

    # Логирование статуса
    logger.log_resource_status()

    # Остановка мониторинга
    logger.stop_resource_monitoring()

    print("✅ Базовый тест мониторинга завершен")


def test_operation_timing():
    """Тест улучшенного форматирования времени операций"""
    print("\n🧪 Тестирование форматирования времени операций...")

    @timed_operation("test_operation")
    def sample_operation():
        """Пример операции для тестирования"""
        print("   Выполнение тестовой операции...")
        time.sleep(2.5)  # Искусственная задержка
        return "Тест завершен"

    # Выполнение операции
    result = sample_operation()
    print(f"   Результат: {result}")

    print("✅ Тест форматирования времени завершен")


def test_resource_alerts():
    """Тест системы оповещений о ресурсах"""
    print("\n🧪 Тестирование системы оповещений...")

    # Установка низких порогов для тестирования
    resource_monitor.set_threshold('cpu_warning', 1.0)  # 1% CPU
    resource_monitor.set_threshold('memory_warning', 1.0)  # 1% памяти

    # Запуск мониторинга
    logger.start_resource_monitoring()

    # Создание нагрузки для генерации оповещений
    print("   Создание нагрузки на CPU...")
    for _ in range(1000000):  # Простая CPU нагрузка
        _ = 2 ** 10

    # Ждем немного для обработки оповещений
    time.sleep(2)

    # Остановка мониторинга
    logger.stop_resource_monitoring()

    print("✅ Тест оповещений завершен")


def test_monitoring_export():
    """Тест экспорта данных мониторинга"""
    print("\n🧪 Тестирование экспорта данных...")

    # Запуск мониторинга
    logger.start_resource_monitoring()

    # Ждем накопления данных
    time.sleep(3)

    # Экспорт истории
    export_path = "test_monitoring_export.json"
    resource_monitor.export_history(export_path)

    # Проверка файла
    if os.path.exists(export_path):
        file_size = os.path.getsize(export_path)
        print(f"   Файл экспорта создан: {export_path} ({file_size} байт)")

        # Удаление тестового файла
        os.remove(export_path)
        print("   Тестовый файл удален")
    else:
        print("   ❌ Файл экспорта не создан")

    # Остановка мониторинга
    logger.stop_resource_monitoring()

    print("✅ Тест экспорта завершен")


def main():
    """Основная функция тестирования"""
    print("🚀 Запуск комплексного тестирования системы мониторинга ресурсов")
    print("=" * 60)

    try:
        # Тест 1: Базовый мониторинг
        test_basic_monitoring()

        # Тест 2: Форматирование времени
        test_operation_timing()

        # Тест 3: Оповещения
        test_resource_alerts()

        # Тест 4: Экспорт данных
        test_monitoring_export()

        print("\n" + "=" * 60)
        print("🎉 Все тесты завершены успешно!")
        print("📊 Система мониторинга ресурсов работает корректно")

    except Exception as e:
        print(f"\n❌ Ошибка во время тестирования: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)