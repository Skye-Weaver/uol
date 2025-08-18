import unittest
import os
import sys
import importlib
from unittest.mock import patch


class TestBuildTranscriptionPrompt(unittest.TestCase):
    def _import_lt(self):
        importlib.invalidate_caches()
        if 'Components.LanguageTasks' in sys.modules:
            importlib.reload(sys.modules['Components.LanguageTasks'])
            return sys.modules['Components.LanguageTasks']
        return importlib.import_module('Components.LanguageTasks')

    def test_build_transcription_prompt_simple_case(self):
        segments = [{"start": 0.0, "end": 1.23, "text": "Привет!"}]
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'dummy'}, clear=False):
            LT = self._import_lt()
            out = LT.build_transcription_prompt(segments)
        self.assertEqual(out, "[0.00] Speaker: Привет! [1.23]\n")
        self.assertTrue(out.endswith("\n"))
        self.assertEqual(out.count("\n"), len(segments))

    def test_build_transcription_prompt_with_speaker_name(self):
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Добро пожаловать!", "speaker": "", "name": "Алиса", "id": "spk1"},
            {"start": 3.0, "end": 4.0, "text": "Сегодня обсудим ИИ."}
        ]
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'dummy'}, clear=False):
            LT = self._import_lt()
            out = LT.build_transcription_prompt(segments)
        lines = out.strip().splitlines()
        self.assertEqual(len(lines), len(segments))
        self.assertTrue(lines[0].startswith("[0.00] Алиса: "))
        self.assertIn("Сегодня обсудим ИИ.", lines[1])


if __name__ == "__main__":
    unittest.main()