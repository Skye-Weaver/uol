from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip
import subprocess
import math
import tempfile
import os
import shlex
import json # For ffprobe output
import numpy as np
import cv2

# Import caption functions from the new module
from .Captions import burn_captions, animate_captions

def extractAudio(video_path):
    try:
        video_clip = VideoFileClip(video_path)
        audio_path = "audio.wav"
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        print(f"Extracted audio to: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"An error occurred while extracting audio: {e}")
        return None


def crop_video(input_file, output_file, start_time, end_time, original_width, original_height):
    """Extracts a video segment using FFmpeg, ensuring original resolution."""
    try:
        duration = end_time - start_time
        if duration <= 0:
            print("Error: End time must be after start time for cropping.")
            return False
            
        ffmpeg_command = [
            'ffmpeg',
            # -ss перед -i: указывает FFmpeg использовать быстрый поиск по ключевым кадрам.
            # Это принципиально для высокой скорости, так как декодируются только необходимые части файла.
            '-ss', str(start_time),
            
            # -i: определяет входной файл.
            '-i', input_file,
            
            # -t: задает длительность сегмента для извлечения.
            '-t', str(duration),
            
            # -map: явно выбирает потоки для включения в выходной файл.
            # 0:v:0 - первый видеопоток из первого входного файла.
            # 0:a:0 - первый аудиопоток из первого входного файла.
            '-map', '0:v:0',
            '-map', '0:a:0',
            
            # -c copy: КЛЮЧЕВОЕ ИЗМЕНЕНИЕ. Эта опция приказывает FFmpeg не перекодировать
            # (decode -> encode), а напрямую копировать данные видео- и аудиопотоков
            # из исходного контейнера в новый. Это операция I/O-bound, быстрая и без потерь качества.
            '-c', 'copy',
            
            # -sn: отключает копирование потоков субтитров, если они присутствуют в исходном файле.
            '-sn',
            
            # -y: перезаписывать выходной файл без интерактивного подтверждения.
            '-y',
            output_file
        ]
        
        print("Running FFmpeg command for segment extraction (crop_video):")
        cmd_string = ' '.join([str(arg) for arg in ffmpeg_command])
        print(f"Command: {cmd_string}")

        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Successfully extracted segment to: {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg during segment extraction: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"An error occurred during segment extraction: {e}")
        return False

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

def crop_bottom_video(input_path, output_path, crop_percentage_bottom):
    """Crops a percentage from the bottom of the video using FFmpeg."""
    try:
        if not 0 < crop_percentage_bottom < 1:
            print("Error: Crop percentage must be between 0 and 1 (exclusive).")
            return False
            
        height_multiplier = 1.0 - crop_percentage_bottom
        
        ffmpeg_command = [
            'ffmpeg',
            '-i', input_path,
            # vf filter: keep original width (iw), calculate new height based on percentage, ensure it's even, crop from top-left (0,0)
            '-vf', f'crop=iw:floor(ih*{height_multiplier}/2)*2:0:0',
            '-c:v', 'libx264', # Re-encode video
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'copy',   # Copy existing audio stream
            '-y',           # Overwrite output file
            output_path
        ]

        print("Running FFmpeg command to crop bottom of video:")
        cmd_string = ' '.join([str(arg) for arg in ffmpeg_command])
        print(f"Command: {cmd_string}")

        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Successfully cropped bottom off video to: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg during bottom crop: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        # Clean up potentially incomplete output file
        if os.path.exists(output_path):
             try: os.remove(output_path) 
             except: pass
        return False
    except Exception as e:
        print(f"An unexpected error occurred during bottom cropping: {e}")
        # Clean up potentially incomplete output file
        if os.path.exists(output_path):
             try: os.remove(output_path) 
             except: pass
        return False

def get_video_dimensions(video_path):
    """Gets the width and height of a video file using ffprobe."""
    try:
        print(f"Checking dimensions for: {video_path}")
        if not os.path.exists(video_path):
            print("  Error: File not found.")
            return None, None
            
        ffprobe_command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0', # Select the first video stream
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(ffprobe_command, check=True, capture_output=True, text=True)
        output_json = json.loads(result.stdout)
        
        if output_json and 'streams' in output_json and len(output_json['streams']) > 0:
            width = output_json['streams'][0].get('width')
            height = output_json['streams'][0].get('height')
            if width is not None and height is not None:
                 print(f"  Dimensions found: {width}x{height}")
                 return int(width), int(height)
            else:
                 print("  Error: Could not find width/height in ffprobe stream data.")
                 return None, None
        else:
            print("  Error: No video streams found by ffprobe or invalid JSON output.")
            print(f"  ffprobe output: {output_json}")
            return None, None
            
    except subprocess.CalledProcessError as e:
        print(f"  Error running ffprobe: {e}")
        print(f"  ffprobe stderr: {e.stderr}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"  Error decoding ffprobe JSON output: {e}")
        if 'result' in locals(): print(f"  Raw ffprobe output: {result.stdout}")
        return None, None
    except Exception as e:
        print(f"  An unexpected error occurred getting dimensions: {e}")
def create_shorts_video(input_path, output_path, crop_width_percentage=0.7, left_crop_percent=0.15, right_crop_percent=0.15):
    """
    Creates a 9:16 "shorts" video from a source video.

    The process involves:
    1. Cropping the central part of the video with specified left and right crop percentages.
    2. Creating a 9:16 canvas.
    3. Placing a blurred version of the original video as the background.
    4. Overlaying the cropped video in the center.

    Args:
        input_path (str): Path to the source video file.
        output_path (str): Path to save the resulting shorts video.
        crop_width_percentage (float): The percentage of the original width to keep (0.0 to 1.0).
            This parameter is kept for backward compatibility but will be ignored if
            left_crop_percent and right_crop_percent are provided.
        left_crop_percent (float): The percentage of the original width to crop from the left side (0.0 to 1.0).
        right_crop_percent (float): The percentage of the original width to crop from the right side (0.0 to 1.0).

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # 1. Получить размеры видео
        original_width, original_height = get_video_dimensions(input_path)
        if not original_width or not original_height:
            print("Error: Could not get video dimensions.")
            return False

        # 2. Рассчитать новые размеры
        target_aspect_ratio = 9 / 16
        
        # Рассчитать ширину кадрирования на основе процентов обрезки слева и справа
        # Если заданы left_crop_percent и right_crop_percent, используем их
        # В противном случае используем crop_width_percentage для обратной совместимости
        if left_crop_percent is not None and right_crop_percent is not None:
            crop_width = int(original_width * (1.0 - left_crop_percent - right_crop_percent))
            # Убедимся, что crop_width положительный
            crop_width = max(1, crop_width)
        else:
            # Использовать crop_width_percentage для обратной совместимости
            crop_width = int(original_width * crop_width_percentage)
        
        # Высота остается прежней
        crop_height = original_height
        
        # Конечная ширина и высота для формата 9:16
        # Конечная высота равна высоте кадрирования
        output_height = crop_height
        # Конечная ширина вычисляется из соотношения 9:16
        output_width = int(output_height * target_aspect_ratio)
        # Убедимся, что ширина четная
        output_width = output_width if output_width % 2 == 0 else output_width + 1

        # 3. Рассчитать позицию обрезки
        # Для новых параметров обрезки слева и справа
        if left_crop_percent is not None and right_crop_percent is not None:
            crop_x = int(original_width * left_crop_percent)
        else:
            # Для обратной совместимости (обрезка по центру)
            crop_x = int((original_width - crop_width) / 2)

        # 4. Собрать команду FFmpeg с filter_complex
        ffmpeg_command = [
            'ffmpeg', '-y', '-i', input_path,
            '-filter_complex',
            (
                # 1. Клонируем видеопоток дважды для фона и основного видео.
                "[0:v]split=2[main][bg];"
                # 2. Обрабатываем фон:
                #    - масштабируем до конечного размера output_width x output_height
                #    - применяем сильное размытие
                f"[bg]scale={output_width}:{output_height},gblur=sigma=25[bg_blurred];"
                # 3. Обрабатываем основное видео:
                #    - обрезаем с учетом процентов обрезки слева и справа
                f"[main]crop={crop_width}:{crop_height}:{crop_x}:0[main_cropped];"
                # 4. Накладываем обрезанное видео на размытый фон:
                #    - (W-w)/2 и (H-h)/2 центрируют наложение
                "[bg_blurred][main_cropped]overlay=(W-w)/2:(H-h)/2"
            ),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '22',
            '-c:a', 'copy', # Простое копирование аудио без перекодирования
            output_path
        ]

        print("Running FFmpeg command to create shorts video:")
        cmd_string = ' '.join(shlex.quote(arg) for arg in ffmpeg_command)
        print(f"Command: {cmd_string}")
        
        # 5. Запустить FFmpeg
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Successfully created shorts video: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg for shorts creation: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during shorts creation: {e}")
        return False

# Example usage:
# if __name__ == "__main__":
#    # ... (old example usage)

