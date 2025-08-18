import unittest
import os
import sys
import json
import importlib
from unittest.mock import patch


class TestBatchMetadata(unittest.TestCase):
    def _import_lt(self):
        importlib.invalidate_caches()
        if 'Components.LanguageTasks' in sys.modules:
            importlib.reload(sys.modules['Components.LanguageTasks'])
            return sys.modules['Components.LanguageTasks']
        return importlib.import_module('Components.LanguageTasks')

    def test_generate_metadata_batch_stubbed_llm(self):
        items = [
            {"id": "seg_1", "text": "О чём видео?"},
            {"id": "seg_2", "text": "Про ИИ и монетизацию"},
        ]

        def make_title(base: str) -> str:
            # Гарантируем длину 40–70 символов
            title = f"{base} — разбор ключевых идей и нюансов"
            if len(title) < 40:
                title = (title + " — детали")[:40]
            if len(title) > 70:
                title = title[:70]
            return title

        class DummyResp:
            def __init__(self, text: str):
                self.text = text

        def stub_call_llm_with_retry(system_instruction, content, generation_config, model=None, max_api_attempts=3):
            # Возвращаем корректный JSON-массив для входных items
            out = []
            for it in items:
                t = make_title(f"{it['text']}".strip())
                desc = "Короткое резюме содержания сегмента без лишних подробностей."
                hashtags = ["#shorts", "#ИИ", "#советы"]
                out.append({
                    "id": it["id"],
                    "title": t,
                    "description": desc[:150],
                    "hashtags": hashtags
                })
            return DummyResp(json.dumps(out, ensure_ascii=False))

        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'dummy'}, clear=False):
            LT = self._import_lt()
            with patch('Components.LanguageTasks.call_llm_with_retry', side_effect=stub_call_llm_with_retry):
                result = LT.generate_metadata_batch(items, max_attempts=1)

        # Проверки
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(items))
        for i, (inp, out) in enumerate(zip(items, result), start=1):
            self.assertEqual(out["id"], inp["id"])
            self.assertIn("title", out)
            self.assertIn("description", out)
            self.assertIn("hashtags", out)
            self.assertIsInstance(out["title"], str)
            self.assertIsInstance(out["description"], str)
            self.assertIsInstance(out["hashtags"], list)
            self.assertGreaterEqual(len(out["title"]), 40)
            self.assertLessEqual(len(out["title"]), 70)
            self.assertLessEqual(len(out["description"]), 150)
            self.assertGreaterEqual(len(out["hashtags"]), 3)
            self.assertLessEqual(len(out["hashtags"]), 5)
            self.assertEqual(out["hashtags"][0], "#shorts")


if __name__ == "__main__":
    unittest.main()