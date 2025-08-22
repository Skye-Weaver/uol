#!/usr/bin/env python3
"""
Скрипт для тестирования загрузки YouTube видео после исправлений.
Используйте этот скрипт для проверки работы обновленного YoutubeDownloader.
"""

import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Components.YoutubeDownloader import download_youtube_video

def test_youtube_download():
    """Тестирование загрузки YouTube видео"""
    print("=== Тестирование загрузки YouTube видео ===\n")

    # Тестовое видео (короткое для быстрой проверки)
    test_urls = [
        "https://www.youtube.com/watch?v=WKe8DvzOhV0",  # Оригинальное видео из ошибки
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Классическое тестовое видео
    ]

    for i, url in enumerate(test_urls, 1):
        print(f"Тест {i}: Загрузка {url}")
        print("-" * 50)

        try:
            result = download_youtube_video(url)
            if result:
                print(f"✅ УСПЕХ: Видео загружено в {result}")
                print(f"   Размер файла: {os.path.getsize(result)} байт")
            else:
                print("❌ НЕУДАЧА: Не удалось загрузить видео")
        except Exception as e:
            print(f"❌ ОШИБКА: {e}")

        print("\n" + "="*60 + "\n")

    # Проверяем наличие cookies файла
    if os.path.exists("cookies.txt"):
        print("✅ Найден файл cookies.txt")
    else:
        print("⚠️  Файл cookies.txt не найден. Рекомендуется добавить его для лучшей работы.")

if __name__ == "__main__":
    test_youtube_download()