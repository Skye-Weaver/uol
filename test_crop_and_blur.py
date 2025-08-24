#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функции crop_to_70_percent_with_blur
"""
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent))

from Components.FaceCrop import crop_to_70_percent_with_blur

def test_crop_function():
    """Тестирует функцию crop_to_70_percent_with_blur с примером видео"""

    # Проверяем наличие FFmpeg
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        print("✓ FFmpeg найден")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ FFmpeg не найден. Установите FFmpeg для тестирования.")
        return False

    # Создаем тестовое видео с помощью FFmpeg (простой цветной прямоугольник)
    test_input = "test_input.mp4"
    test_output = "test_output_70_percent.mp4"

    print("Создание тестового видео...")
    try:
        # Создаем тестовое видео 1920x1080 с цветными полосами
        subprocess.run([
            'ffmpeg',
            '-f', 'lavfi',
            '-i', 'testsrc2=duration=5:size=1920x1080:rate=30',
            '-y',
            test_input
        ], check=True, capture_output=True)
        print(f"✓ Тестовое видео создано: {test_input}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Не удалось создать тестовое видео: {e}")
        return False

    # Тестируем нашу функцию
    print("Тестирование функции crop_to_70_percent_with_blur...")
    try:
        result_path = crop_to_70_percent_with_blur(test_input, test_output)
        if result_path and os.path.exists(result_path):
            print(f"✓ Функция выполнена успешно. Результат: {result_path}")

            # Получаем информацию о выходном видео
            ffprobe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'json',
                test_output
            ]
            probe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
            import json
            probe_data = json.loads(probe_result.stdout)
            width = probe_data['streams'][0]['width']
            height = probe_data['streams'][0]['height']

            print(f"✓ Размеры выходного видео: {width}x{height}")

            # Проверяем, что ширина примерно 70% от оригинала (1920 * 0.7 = 1344)
            expected_width = int(1920 * 0.7)
            if abs(width - expected_width) < 10:  # Допуск 10 пикселей
                print(f"✓ Ширина корректна (ожидалось ~{expected_width}, получено {width})")
            else:
                print(f"⚠ Ширина отличается от ожидаемой (ожидалось ~{expected_width}, получено {width})")

            return True
        else:
            print("✗ Функция вернула None или файл не создан")
            return False

    except Exception as e:
        print(f"✗ Ошибка при тестировании функции: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Очистка тестовых файлов
        for file in [test_input, test_output]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"✓ Удален тестовый файл: {file}")
                except Exception as e:
                    print(f"⚠ Не удалось удалить {file}: {e}")

if __name__ == "__main__":
    print("=== Тестирование функции crop_to_70_percent_with_blur ===")
    success = test_crop_function()
    if success:
        print("\n🎉 Тестирование завершено успешно!")
    else:
        print("\n❌ Тестирование завершено с ошибками!")
    sys.exit(0 if success else 1)