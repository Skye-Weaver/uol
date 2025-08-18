import unittest
import os
import sys
import importlib
import io
import contextlib
from unittest.mock import patch


class TestRateLimitHandling(unittest.TestCase):
    def _import_lt(self):
        importlib.invalidate_caches()
        if 'Components.LanguageTasks' in sys.modules:
            importlib.reload(sys.modules['Components.LanguageTasks'])
            return sys.modules['Components.LanguageTasks']
        return importlib.import_module('Components.LanguageTasks')

    def test_rate_limit_then_success(self):
        # Первый вызов — ResourceExhausted с retryDelay=1s, второй — успешный ответ "[]"
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'dummy'}, clear=False):
            LT = self._import_lt()
            gen_cfg = LT.make_generation_config("sys-prompt")

            class DummyResp:
                def __init__(self, text: str):
                    self.text = text

            with patch('Components.LanguageTasks.time.sleep', return_value=None) as p_sleep:
                with patch('Components.LanguageTasks.client.models.generate_content',
                           side_effect=[Exception("ResourceExhausted: retryDelay: 1s"), DummyResp("[]")]) as p_gen:
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        resp = LT.call_llm_with_retry(
                            system_instruction=None,
                            content=[],  # Передаём список, чтобы внутри не создавались types.Content
                            generation_config=gen_cfg,
                            model=None,
                            max_api_attempts=3,
                        )
                    log = buf.getvalue()

        self.assertIsNotNone(resp)
        self.assertEqual(getattr(resp, "text", None), "[]")
        # Проверяем, что был лог про паузу на 1 секунд перед попыткой #2
        self.assertIn("Лимит API обработан. Выполняю паузу на 1 секунд перед попыткой #2.", log)

    def test_rate_limit_without_retry_delay(self):
        # Один вызов — ResourceExhausted без retryDelay; ожидаем re-raise и соответствующий лог
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'dummy'}, clear=False):
            LT = self._import_lt()
            gen_cfg = LT.make_generation_config("sys-prompt")

            with patch('Components.LanguageTasks.time.sleep', return_value=None):
                with patch('Components.LanguageTasks.client.models.generate_content',
                           side_effect=Exception("ResourceExhausted: quota exceeded")):
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        with self.assertRaises(Exception):
                            LT.call_llm_with_retry(
                                system_instruction=None,
                                content=[],
                                generation_config=gen_cfg,
                                model=None,
                                max_api_attempts=3,
                            )
                    log = buf.getvalue()

        self.assertIn("Не удалось извлечь retryDelay. Попытки прекращены.", log)


if __name__ == "__main__":
    unittest.main()