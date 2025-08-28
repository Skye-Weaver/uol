#!/usr/bin/env python3
"""
Тестовый сценарий для проверки функциональности режима "фильм"
Проверяет интеграцию компонентов и базовую логику без запуска полного анализа.
"""

import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь для импортов
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Тест импортов всех необходимых компонентов"""
    print("=== Тест импортов ===")
    try:
        from Components.FilmMode import FilmAnalyzer, analyze_film_main, FilmMoment, RankedMoment
        from Components.config import get_config, FilmModeConfig
        from Components.YoutubeDownloader import download_youtube_video
        from Components.Transcription import transcribe_unified
        from Components.LanguageTasks import build_transcription_prompt, call_llm_with_retry, make_generation_config
        from Components.Database import VideoDatabase
        from Components.Logger import logger

        print("✓ Все импорты успешны")
        return True
    except Exception as e:
        print(f"✗ Ошибка импорта: {e}")
        return False

def test_config_loading():
    """Тест загрузки конфигурации режима фильм"""
    print("\n=== Тест загрузки конфигурации ===")
    try:
        from Components.config import get_config
        config = get_config()

        # Проверяем наличие film_mode в конфигурации
        if hasattr(config, 'film_mode'):
            film_config = config.film_mode
            print(f"✓ Режим фильм включен: {film_config.enabled}")
            print(f"✓ Максимум моментов: {film_config.max_moments}")
            print(f"✓ Длительности COMBO: {film_config.combo_duration}")
            print(f"✓ Длительности SINGLE: {film_config.single_duration}")
            print(f"✓ Порог паузы: {film_config.pause_threshold}")
            print(f"✓ Модель LLM: {film_config.llm_model}")
            return True
        else:
            print("✗ film_mode не найден в конфигурации")
            return False
    except Exception as e:
        print(f"✗ Ошибка загрузки конфигурации: {e}")
        return False

def test_film_analyzer_initialization():
    """Тест инициализации FilmAnalyzer"""
    print("\n=== Тест инициализации FilmAnalyzer ===")
    try:
        from Components.FilmMode import FilmAnalyzer
        from Components.config import get_config

        config = get_config()
        analyzer = FilmAnalyzer(config)

        # Проверяем наличие необходимых атрибутов
        assert hasattr(analyzer, 'config'), "Отсутствует атрибут config"
        assert hasattr(analyzer, 'film_config'), "Отсутствует атрибут film_config"
        assert analyzer.film_config is not None, "film_config не инициализирован"

        print("✓ FilmAnalyzer успешно инициализирован")
        print(f"✓ Конфигурация фильма: max_moments={analyzer.film_config.max_moments}")
        return True
    except Exception as e:
        print(f"✗ Ошибка инициализации FilmAnalyzer: {e}")
        return False

def test_film_moment_creation():
    """Тест создания FilmMoment объектов"""
    print("\n=== Тест создания FilmMoment ===")
    try:
        from Components.FilmMode import FilmMoment

        # Создаем тестовый момент
        moment = FilmMoment(
            moment_type="COMBO",
            start_time=10.5,
            end_time=25.3,
            text="Тестовый текст момента",
            context="Тестовый контекст"
        )

        assert moment.moment_type == "COMBO", "Неверный тип момента"
        assert moment.start_time == 10.5, "Неверное время начала"
        assert moment.end_time == 25.3, "Неверное время окончания"
        assert moment.text == "Тестовый текст момента", "Неверный текст"
        assert moment.context == "Тестовый контекст", "Неверный контекст"

        print("✓ FilmMoment успешно создан и проверен")
        return True
    except Exception as e:
        print(f"✗ Ошибка создания FilmMoment: {e}")
        return False

def test_ranking_weights():
    """Тест весовых коэффициентов ранжирования"""
    print("\n=== Тест весовых коэффициентов ===")
    try:
        from Components.config import get_config

        config = get_config()
        weights = config.film_mode.ranking_weights

        # Проверяем наличие всех необходимых весов
        required_weights = [
            'emotional_peaks', 'conflict_escalation', 'punchlines_wit',
            'quotability_memes', 'stakes_goals', 'hooks_cliffhangers', 'visual_penalty'
        ]

        for weight_name in required_weights:
            assert weight_name in weights, f"Отсутствует вес: {weight_name}"
            assert isinstance(weights[weight_name], (int, float)), f"Вес {weight_name} не является числом"

        print("✓ Все весовые коэффициенты присутствуют и корректны")
        print(f"✓ Сумма положительных весов: {sum(v for k, v in weights.items() if v > 0):.2f}")
        return True
    except Exception as e:
        print(f"✗ Ошибка проверки весовых коэффициентов: {e}")
        return False

def test_error_handling():
    """Тест обработки ошибок"""
    print("\n=== Тест обработки ошибок ===")
    try:
        from Components.FilmMode import analyze_film_main

        # Тест с некорректными параметрами
        result = analyze_film_main(url=None, local_path=None)

        # Проверяем, что возвращается корректный результат ошибки
        assert hasattr(result, 'video_id'), "Результат не содержит video_id"
        assert result.video_id == "error", "Неожиданный video_id при ошибке"
        assert "Ошибка анализа" in result.preview_text, "Некорректный текст ошибки"

        print("✓ Обработка ошибок работает корректно")
        return True
    except Exception as e:
        print(f"✗ Ошибка в тесте обработки ошибок: {e}")
        return False

def test_main_menu_integration():
    """Тест интеграции с главным меню main.py"""
    print("\n=== Тест интеграции с главным меню ===")
    try:
        # Проверяем, что функция analyze_film_main доступна для импорта
        from Components.FilmMode import analyze_film_main

        # Проверяем сигнатуру функции
        import inspect
        sig = inspect.signature(analyze_film_main)
        params = list(sig.parameters.keys())

        assert 'url' in params, "Функция не принимает параметр url"
        assert 'local_path' in params, "Функция не принимает параметр local_path"

        print("✓ Функция analyze_film_main корректно интегрирована")
        print(f"✓ Параметры функции: {params}")
        return True
    except Exception as e:
        print(f"✗ Ошибка интеграции с главным меню: {e}")
        return False

def run_all_tests():
    """Запуск всех тестов"""
    print("🚀 Запуск тестов режима 'фильм'\n")

    tests = [
        test_imports,
        test_config_loading,
        test_film_analyzer_initialization,
        test_film_moment_creation,
        test_ranking_weights,
        test_error_handling,
        test_main_menu_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Тест {test.__name__} завершился с исключением: {e}")

    print(f"\n{'='*50}")
    print(f"📊 Результаты тестирования: {passed}/{total} тестов пройдено")

    if passed == total:
        print("🎉 Все тесты пройдены! Режим 'фильм' готов к использованию.")
        return True
    else:
        print(f"⚠️  {total - passed} тест(ов) не пройдено. Проверьте логи выше.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)