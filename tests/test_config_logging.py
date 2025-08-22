#!/usr/bin/env python3
"""
Тест для проверки загрузки конфигурации с секцией logging.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Components.config import get_config, AppConfig

def test_config_loading():
    """Тестируем загрузку конфигурации с секцией logging."""
    print("🧪 Тестирование загрузки конфигурации...")

    try:
        # Загружаем конфигурацию
        cfg = get_config()

        # Проверяем, что это объект AppConfig
        assert isinstance(cfg, AppConfig), f"Ожидался AppConfig, получен {type(cfg)}"

        # Проверяем наличие секции logging
        assert hasattr(cfg, 'logging'), "AppConfig не содержит секцию logging"

        # Проверяем основные поля logging
        logging_config = cfg.logging
        assert hasattr(logging_config, 'log_dir'), "LoggingConfig не содержит log_dir"
        assert hasattr(logging_config, 'log_level'), "LoggingConfig не содержит log_level"
        assert hasattr(logging_config, 'enable_performance_monitoring'), "LoggingConfig не содержит enable_performance_monitoring"
        assert hasattr(logging_config, 'enable_gpu_monitoring'), "LoggingConfig не содержит enable_gpu_monitoring"
        assert hasattr(logging_config, 'enable_progress_bars'), "LoggingConfig не содержит enable_progress_bars"
        assert hasattr(logging_config, 'gpu_priority_mode'), "LoggingConfig не содержит gpu_priority_mode"

        # Проверяем значения по умолчанию
        assert logging_config.log_dir == "logs", f"Ожидалось 'logs', получено '{logging_config.log_dir}'"
        assert logging_config.log_level == "INFO", f"Ожидалось 'INFO', получено '{logging_config.log_level}'"
        assert logging_config.enable_performance_monitoring == True, f"Ожидалось True, получено {logging_config.enable_performance_monitoring}"
        assert logging_config.enable_gpu_monitoring == True, f"Ожидалось True, получено {logging_config.enable_gpu_monitoring}"
        assert logging_config.enable_progress_bars == True, f"Ожидалось True, получено {logging_config.enable_progress_bars}"
        assert logging_config.gpu_priority_mode == True, f"Ожидалось True, получено {logging_config.gpu_priority_mode}"

        print("✅ Все проверки конфигурации пройдены!")
        print(f"📊 log_dir: {logging_config.log_dir}")
        print(f"📊 log_level: {logging_config.log_level}")
        print(f"📊 enable_performance_monitoring: {logging_config.enable_performance_monitoring}")
        print(f"📊 enable_gpu_monitoring: {logging_config.enable_gpu_monitoring}")
        print(f"📊 enable_progress_bars: {logging_config.enable_progress_bars}")
        print(f"📊 gpu_priority_mode: {logging_config.gpu_priority_mode}")

        return True

    except Exception as e:
        print(f"❌ Ошибка при тестировании конфигурации: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)