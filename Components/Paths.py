from typing import Tuple
import os
from pathlib import Path
from Components.config import get_config

def build_short_output_name(base_name: str, idx: int, shorts_dir: str) -> Tuple[str, str]:
    """
    Формирует уникальные пути для итогового short-файла и соответствующего временного файла анимации.

    Итоговый шаблон:
    shorts/{base_name}_highlight_{idx:02d}_final.mp4

    Временный файл для анимации:
    {final_path}_temp_anim.mp4
    """
    file_name = f"{base_name}_highlight_{idx:02d}_final.mp4"
    final_path = os.path.join(shorts_dir, file_name)
    temp_anim_path = f"{final_path}_temp_anim.mp4"
    return final_path, temp_anim_path


def resolve_path(*parts: str) -> str:
    """
    Возвращает абсолютный путь, собранный относительно base_dir из конфига.

    Пример:
      resolve_path("a", "b", "c.txt") -> "<ABS_BASE_DIR>/a/b/c.txt"

    Всегда нормализует и резолвит путь (Path(...).resolve()).
    """
    cfg = get_config()
    base_dir = Path(cfg.paths.base_dir)
    abs_path = (base_dir.joinpath(*[str(p) for p in parts])).resolve()
    return str(abs_path)


def fonts_path(font_filename: str) -> str:
    """
    Возвращает абсолютный путь к шрифту, собранный от base_dir/fonts_dir.

    Пример:
      fonts_path("Montserrat-Bold.ttf") -> "<ABS_BASE_DIR>/<fonts_dir>/Montserrat-Bold.ttf"

    Всегда нормализует и резолвит путь (Path(...).resolve()).
    """
    cfg = get_config()
    base_dir = Path(cfg.paths.base_dir)
    fonts_root = (base_dir / cfg.paths.fonts_dir).resolve()
    return str((fonts_root / font_filename).resolve())