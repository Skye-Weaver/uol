import pytest

from main import find_last_word_end_time, prepare_words_for_segment


def _make_transcription(words_list):
    return {"segments": [{"words": words_list}]}


def test_find_last_word_end_time_extends_stop():
    # Последнее слово начинается до segment_end=32.5 (start=32.3) и заканчивается после (end=32.84)
    transcription = _make_transcription([
        {"start": 31.0, "end": 31.4, "text": "alpha"},
        {"start": 32.3, "end": 32.84, "text": "omega"},
        {"start": 32.6, "end": 32.9, "text": "ignored_late"},
    ])
    segment_end = 32.5
    last_end = find_last_word_end_time(transcription, segment_end)
    assert last_end == pytest.approx(32.84, abs=1e-6)

    original_stop = 32.5
    adjusted_stop = max(original_stop, last_end if last_end is not None else original_stop)
    assert adjusted_stop == pytest.approx(32.84, abs=1e-6)


def test_find_last_word_end_time_no_words():
    # Пустые сегменты
    transcription_empty = {"segments": []}
    assert find_last_word_end_time(transcription_empty, 33.0) is None

    # Нерелевантные слова (start >= segment_end_time)
    transcription_irrelevant = _make_transcription([
        {"start": 33.5, "end": 33.9, "text": "later"},
        {"start": 40.0, "end": 41.0, "text": "much_later"},
    ])
    assert find_last_word_end_time(transcription_irrelevant, 33.0) is None

    # Некорректные таймкоды (нечисловые) — должны пропускаться и давать None
    transcription_bad = _make_transcription([
        {"start": None, "end": 32.0, "text": "bad1"},
        {"start": "x", "end": "y", "text": "bad2"},
    ])
    assert find_last_word_end_time(transcription_bad, 33.0) is None


def test_prepare_words_for_segment_normalizes_times():
    start = 30.0
    stop = 33.0  # Длительность сегмента = 3.0
    transcription = _make_transcription([
        {"start": 30.1, "end": 30.5, "text": "a"},
        {"start": 32.0, "end": 33.2, "text": "b"},   # выйдет за пределы и будет обрезано до 3.0
        {"start": 29.5, "end": 29.9, "text": "out_before"},
        {"start": 33.1, "end": 33.4, "text": "out_after"},
    ])

    result = prepare_words_for_segment(transcription, start, stop)
    segs = result.get("segments", [])
    assert len(segs) == 1
    seg = segs[0]
    assert seg.get("start") == pytest.approx(0.0, abs=1e-6)
    assert seg.get("end") == pytest.approx(3.0, abs=1e-6)

    words = seg.get("words", [])
    # Ожидаем только 2 слова: (30.1-30.5) и (32.0-33.2->clip 33.0)
    assert len(words) == 2

    # Сортировка по start_rel
    w0, w1 = words[0], words[1]
    # Первое слово
    assert w0["text"] == "a"
    assert w0["start"] == pytest.approx(0.1, abs=1e-6)
    assert w0["end"] == pytest.approx(0.5, abs=1e-6)
    # Второе слово
    assert w1["text"] == "b"
    assert w1["start"] == pytest.approx(2.0, abs=1e-6)
    assert w1["end"] == pytest.approx(3.0, abs=1e-6)  # обрезано по границе сегмента


def test_prepare_words_for_segment_overlap_filtering():
    start = 30.0
    stop = 33.0
    transcription = _make_transcription([
        {"start": 29.7, "end": 30.2, "text": "cross_start"},  # пересекает начало (clamp к 0.0..0.2)
        {"start": 32.9, "end": 33.5, "text": "cross_end"},    # пересекает конец (end clip к 3.0)
        {"start": 28.0, "end": 29.6, "text": "outside_before"},  # полностью вне
    ])

    result = prepare_words_for_segment(transcription, start, stop)
    segs = result.get("segments", [])
    assert len(segs) == 1
    seg = segs[0]
    assert seg.get("end") == pytest.approx(3.0, abs=1e-6)

    words = seg.get("words", [])
    # Должны остаться только два пересекающихся слова
    assert len(words) == 2

    # Проверяем корректные сдвиги/обрезки и порядок
    w0, w1 = words[0], words[1]
    assert w0["text"] == "cross_start"
    assert w0["start"] == pytest.approx(0.0, abs=1e-6)
    assert w0["end"] == pytest.approx(0.2, abs=1e-6)

    assert w1["text"] == "cross_end"
    assert w1["start"] == pytest.approx(2.9, abs=1e-6)
    assert w1["end"] == pytest.approx(3.0, abs=1e-6)