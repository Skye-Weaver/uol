import math
import cv2
import numpy as np # Add numpy import
import os
import subprocess
import traceback # For detailed error printing in animate_captions
from PIL import Image, ImageDraw, ImageFont # Pillow imports for custom font

# Function to format time in SRT format
def format_time(seconds):
    milliseconds = int((seconds - math.floor(seconds)) * 1000)
    seconds = int(math.floor(seconds))
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Function to format time in ASS format (hours:mm:ss.cc)
def format_time_ass(seconds):
    centiseconds = int((seconds - math.floor(seconds)) * 100)
    seconds = int(math.floor(seconds))
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

from .config import get_config

def create_ass_file(
    word_level_transcription: dict,
    output_ass_path: str,
    video_width: int,
    video_height: int,
    segment_start_time: float,
    segment_end_time: float
) -> bool:
    """
    Создает .ass файл с анимированными субтитрами (эффект караоке).
    """
    cfg = get_config()
    try:
        # --- Стили ---
        # Рассчитываем размер шрифта относительно высоты видео
        font_size = int(video_height / cfg.captions.font_size_ratio)
        outline = int(font_size / cfg.captions.outline_ratio)
        shadow = int(outline / cfg.captions.shadow_ratio)
        margin_v = int(video_height / cfg.captions.margin_v_ratio)

        ass_header = f"""[Script Info]
Title: AI Generated Captions
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{cfg.captions.font_name}, {font_size},&H00FFFFFF,&H000000FF,&H00000000,&H60000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        with open(output_ass_path, 'w', encoding='utf-8') as f:
            f.write(ass_header)
            
            all_words = []
            if "segments" in word_level_transcription:
                for segment in word_level_transcription["segments"]:
                    if "words" in segment:
                        all_words.extend(segment["words"])

            # Фильтруем слова, которые попадают в наш временной отрезок
            segment_words = [
                word for word in all_words
                if word.get("start") is not None and word.get("end") is not None and
                   word["start"] < segment_end_time and word["end"] > segment_start_time
            ]

            if not segment_words:
                print("No words found for the given time segment.")
                return True # Файл создан, хоть и пустой

            # Группируем слова в строки (например, по 2-3 слова)
            line_buffer = []
            line_start_time = -1

            for i, word_info in enumerate(segment_words):
                word_text = str(word_info.get("word", word_info.get("text", ""))).strip()
                if not word_text:
                    continue

                if line_start_time < 0:
                    line_start_time = word_info["start"]

                line_buffer.append(word_info)

                # Записываем строку, если она содержит 2 слова или это последнее слово
                if len(line_buffer) >= 2 or i == len(segment_words) - 1:
                    line_end_time = line_buffer[-1]["end"]
                    
                    # Нормализуем время относительно начала сегмента
                    relative_line_start = max(0, line_start_time - segment_start_time)
                    relative_line_end = line_end_time - segment_start_time

                    # Собираем текст строки и тайминги для караоке
                    full_line_text = ""
                    karaoke_tags = ""
                    current_duration = 0

                    for j, word in enumerate(line_buffer):
                        start_rel = max(0, word["start"] - line_start_time)
                        end_rel = word["end"] - line_start_time
                        
                        duration_ms = int((end_rel - start_rel) * 100)
                        
                        # Добавляем \k с задержкой, если это не первое слово
                        if j > 0:
                            delay = int((word["start"] - line_buffer[j-1]["end"]) * 100)
                            karaoke_tags += f"\\k{delay}"

                        karaoke_tags += f"\\k{duration_ms}"
                        full_line_text += str(word.get("word", word.get("text", ""))).strip() + " "
                    
                    dialogue_line = (
                        f"Dialogue: 0,{format_time_ass(relative_line_start)},{format_time_ass(relative_line_end)},"
                        f"Default,,0,0,0,,{{ {karaoke_tags.strip()} }}{full_line_text.strip()}"
                    )
                    f.write(dialogue_line + "\\N")

                    # Сбрасываем буфер
                    line_buffer = []
                    line_start_time = -1

        return True

    except Exception as e:
        print(f"Error creating ASS file: {e}")
        traceback.print_exc()
        return False

# Function to burn captions using FFmpeg (DEPRECATED but kept for reference/fallback)
def burn_captions(vertical_video_path, audio_source_path, transcriptions, start_time, end_time, output_path):
    print("[WARN] burn_captions is deprecated. Use the new single-pass pipeline.")
    return False


# --- Word-Level Animation Helpers ---

def find_active_segment_and_word(transcription_result, current_time):
    """Finds the segment and word active at the current time."""
    active_segment = None
    active_word_index_in_segment = -1 # Index within the segment's word list

    for segment in transcription_result.get("segments", []):
        # Use segment boundaries to find the active segment
        if segment['start'] <= current_time < segment['end']:
            active_segment = segment
            # Find the specific word within this segment based on word timings
            for i, word_info in enumerate(segment.get("words", [])):
                # Ensure word timings exist before comparing
                if 'start' in word_info and 'end' in word_info:
                    if word_info['start'] <= current_time < word_info['end']:
                        active_word_index_in_segment = i
                        break # Found the active word
            # If no specific word is active but the segment is, keep active_segment
            # active_word_index_in_segment will remain -1 or the found index
            break # Found the active segment

    return active_segment, active_word_index_in_segment


# --- Main Animation Function ---
