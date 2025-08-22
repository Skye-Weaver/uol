import os
import json

from Components.Transcription import export_transcription_artifacts


def test_export_transcription_artifacts(tmp_path):
    """
    Юнит-тест экспортера транскрипции:
    - пишет TXT/JSON/SRT/VTT в заданный каталог
    - проверяет базовую валидность каждого формата
    """
    # Подготовка фиктивной транскрипции
    word_level_transcription = {
        "segments": [
            {
                "start": 0.12,
                "end": 1.34,
                "text": "Привет, мир!",
                "words": [
                    {"word": "Привет,", "start": 0.12, "end": 0.60},
                    {"word": "мир!", "start": 0.61, "end": 1.10},
                ],
            },
            {
                "start": 2.00,
                "end": 3.50,
                "text": "Это тестовая фраза.",
                "words": [
                    {"word": "Это", "start": 2.00, "end": 2.20},
                    {"word": "тестовая", "start": 2.21, "end": 3.00},
                    {"word": "фраза.", "start": 3.01, "end": 3.50},
                ],
            },
        ]
    }

    base_name = "unit_test_transcription"
    out_dir = tmp_path / "exports"

    # Вызов экспортера
    paths = export_transcription_artifacts(base_name, word_level_transcription, str(out_dir))

    # Проверяем существование всех файлов
    txt_path = paths["txt"]
    json_path = paths["json"]
    srt_path = paths["srt"]
    vtt_path = paths["vtt"]

    assert os.path.exists(txt_path), "TXT файл не создан"
    assert os.path.exists(json_path), "JSON файл не создан"
    assert os.path.exists(srt_path), "SRT файл не создан"
    assert os.path.exists(vtt_path), "VTT файл не создан"

    # TXT: файл непустой и содержит текст сегмента
    with open(txt_path, "r", encoding="utf-8") as f:
        txt_content = f.read()
    assert len(txt_content.strip()) > 0, "TXT файл пустой"
    assert "Привет, мир!" in txt_content, "TXT не содержит ожидаемый текст"

    # JSON: корректно парсится и содержит ключ 'segments'
    with open(json_path, "r", encoding="utf-8") as f:
        json_content = f.read()
    data = json.loads(json_content)
    assert "segments" in data, "JSON не содержит ключ 'segments'"
    assert isinstance(data["segments"], list) and len(data["segments"]) > 0, "JSON 'segments' пустой"

    # SRT: содержит таймкод и индексы
    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()
    assert "-->" in srt_content.replace(" ", ""), "SRT не содержит разделителя времени '-->'"
    first_line = srt_content.strip().splitlines()[0]
    assert first_line.isdigit() and int(first_line) == 1, "SRT не начинается с индекса 1"

    # VTT: начинается с 'WEBVTT'
    with open(vtt_path, "r", encoding="utf-8") as f:
        vtt_content = f.read()
    assert vtt_content.startswith("WEBVTT"), "VTT не начинается с заголовка 'WEBVTT'"