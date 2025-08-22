#!/usr/bin/env python3
"""
Тестовый файл для проверки интеграции логирования в функции транскрипции.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Components.Transcription import _log_system_resources
from Components.Logger import logger

def test_system_resources_logging():
    """Тестирует логирование системных ресурсов."""
    print("=== Тестирование логирования системных ресурсов ===")

    try:
        # Тестируем логирование системных ресурсов
        _log_system_resources("Тестовый запуск")
        print("✓ Логирование системных ресурсов работает")
        return True
    except Exception as e:
        print(f"✗ Ошибка при логировании системных ресурсов: {e}")
        return False

def test_transcription_function_import():
    """Тестирует импорт функции транскрипции."""
    print("\n=== Тестирование импорта функции транскрипции ===")

    try:
        from Components.Transcription import transcribe_unified
        print("✓ Функция transcribe_unified успешно импортирована")
        print(f"✓ Функция имеет декоратор: {hasattr(transcribe_unified, '__wrapped__')}")
        return True
    except Exception as e:
        print(f"✗ Ошибка при импорте transcribe_unified: {e}")
        return False

def test_logger_integration():
    """Тестирует интеграцию с системой логирования."""
    print("\n=== Тестирование интеграции с системой логирования ===")

    try:
        # Тестируем базовые функции логгера
        logger.logger.info("Тестовое сообщение в лог")
        logger.logger.debug("Тестовое debug сообщение")

        # Проверяем наличие функции create_progress_bar
        if hasattr(logger, 'create_progress_bar'):
            print("✓ Функция create_progress_bar доступна")
        else:
            print("✗ Функция create_progress_bar не найдена")
            return False

        print("✓ Интеграция с системой логирования работает")
        return True
    except Exception as e:
        print(f"✗ Ошибка при тестировании интеграции логирования: {e}")
        return False

def main():
    """Основная функция тестирования."""
    print("Запуск тестов интеграции логирования в систему транскрипции\n")

    tests = [
        test_system_resources_logging,
        test_transcription_function_import,
        test_logger_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n=== Результаты тестирования ===")
    print(f"Пройдено: {passed}/{total}")

    if passed == total:
        print("🎉 Все тесты пройдены! Интеграция логирования готова к использованию.")
        return 0
    else:
        print("⚠️  Некоторые тесты не пройдены. Проверьте ошибки выше.")
        return 1

if __name__ == "__main__":
    exit(main())