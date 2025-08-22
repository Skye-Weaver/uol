import os
from Components.Paths import build_short_output_name

def test_build_short_output_name_unique_and_temp_suffix():
    base = "master-mUhy4zXaCn8"
    shorts_dir = "shorts"

    final1, temp1 = build_short_output_name(base, 1, shorts_dir)
    final2, temp2 = build_short_output_name(base, 2, shorts_dir)

    # Проверяем zero-pad индекса и директорию
    assert final1 == os.path.join(shorts_dir, f"{base}_highlight_01_final.mp4")
    assert final2 == os.path.join(shorts_dir, f"{base}_highlight_02_final.mp4")

    # Проверяем уникальность имён
    assert final1 != final2

    # Проверяем формирование временного файла анимации
    assert temp1 == final1 + "_temp_anim.mp4"
    assert temp2 == final2 + "_temp_anim.mp4"