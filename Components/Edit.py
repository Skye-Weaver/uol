from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip
import subprocess
import math
import tempfile
import os
import shlex
import json # For ffprobe output

import cv2
import numpy as np

# Import caption functions from the new module
from .Captions import burn_captions, animate_captions, create_ass_file

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
        return None, None

# Example usage:
# if __name__ == "__main__":
#    # ... (old example usage)


def process_frame_for_vertical_short(
    source_video_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    word_level_transcription: dict,
    crop_bottom_percent: float = 0.0,
    face_cascade_path: str = 'haarcascade_frontalface_default.xml'
) -> bool:
    """
    Оптимизированный однопроходный конвейер для создания вертикального видео.

    Выполняет следующие операции за один вызов ffmpeg:
    1.  Извлечение сегмента видео (`-ss`, `-to`).
    2.  Анализ лиц для определения центра кадрирования.
    3.  Динамическое кадрирование в вертикальный формат (9:16) с центрированием на лицах.
    4.  Опциональная обрезка нижней части видео.
    5.  Наложение анимированных субтитров через ASS-файл.
    6.  Копирование аудиодорожки из исходного сегмента.

    Возвращает True в случае успеха, иначе False.
    """
    temp_dir = None
    ass_file_path = None
    try:
        # 1. Проверка входных данных
        if not os.path.exists(source_video_path):
            print(f"Error: Source video not found at '{source_video_path}'")
            return False
        if not os.path.exists(face_cascade_path):
            print(f"Error: Face cascade classifier not found at '{face_cascade_path}'")
            return False
        if end_time <= start_time:
            print("Error: End time must be greater than start time.")
            return False

        # 2. Получение размеров видео
        original_width, original_height = get_video_dimensions(source_video_path)
        if not original_width or not original_height:
            print("Error: Could not get video dimensions.")
            return False

        # 3. Анализ лиц для определения средней позиции
        print("Analyzing face positions for smart cropping...")
        cap = cv2.VideoCapture(source_video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        face_positions = []
        
        frame_count = 0
        while cap.isOpened():
            current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time_sec > end_time:
                break
            
            ret, frame = cap.read()
            if not ret:
                break

            # Анализируем каждый N-й кадр для производительности
            if frame_count % 3 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    # Усредняем позиции всех найденных лиц в кадре
                    avg_x = np.mean([x + w / 2 for x, y, w, h in faces])
                    face_positions.append(avg_x)
            frame_count += 1
        
        cap.release()

        # Вычисляем среднюю позицию лица по всему клипу
        if face_positions:
            average_face_x = int(np.mean(face_positions))
        else:
            # Если лиц не найдено, центрируемся по горизонтали
            average_face_x = original_width // 2
        
        print(f"Average horizontal face position: {average_face_x}px")

        # 4. Расчет параметров кадрирования
        target_height = original_height
        if crop_bottom_percent > 0:
            target_height = int(original_height * (1.0 - crop_bottom_percent))

        # Соотношение сторон 9:16
        target_width = int(target_height * 9 / 16)

        # Гарантируем, что ширина кадрирования четная (требование многих кодеков)
        if target_width % 2 != 0:
            target_width += 1
        
        # Определяем x_offset для кадрирования, центрируясь на average_face_x
        crop_x = max(0, average_face_x - target_width // 2)
        # Убеждаемся, что не выходим за правую границу
        if crop_x + target_width > original_width:
            crop_x = original_width - target_width
        
        # 5. Создание временного ASS-файла для субтитров
        print("Preparing animated captions...")
        temp_dir = tempfile.mkdtemp()
        ass_file_path = os.path.join(temp_dir, "captions.ass")

        # Эта функция должна быть адаптирована или импортирована, она создает ASS файл
        # на основе словной транскрипции.
        ass_success = create_ass_file(
            word_level_transcription,
            ass_file_path,
            target_width,
            target_height,
            start_time,
            end_time
        )
        if not ass_success:
            print("Error: Failed to create ASS subtitle file.")
            # Не возвращаем False, можем продолжить без субтитров
            ass_file_path = None

        # 6. Сборка и выполнение команды FFmpeg
        print("Building and running final FFmpeg command...")
        
        # Используем `to` вместо `t` для точности
        duration = end_time - start_time
        
        ffmpeg_command = [
            'ffmpeg',
            '-ss', str(start_time),   # Точное время начала
            '-to', str(end_time),     # Точное время окончания
            '-i', source_video_path,
            '-c:a', 'copy',           # Копируем аудио без перекодирования
        ]

        # Комплексный фильтр
        video_filters = []
        # 1. Кадрирование: crop=width:height:x:y
        video_filters.append(f"crop={target_width}:{target_height}:{crop_x}:0")
        
        # 2. Наложение субтитров, если они есть
        if ass_file_path and os.path.exists(ass_file_path):
            # Важно: путь к файлу нужно правильно экранировать для ffmpeg
            escaped_ass_path = ass_file_path.replace('\\', '/').replace(':', '\\\\:')
            video_filters.append(f"ass='{escaped_ass_path}'")
        
        if video_filters:
            ffmpeg_command.extend(['-vf', ",".join(video_filters)])

        # Параметры видеокодека
        ffmpeg_command.extend([
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-aspect', '9:16', # Устанавливаем правильное соотношение сторон
            '-y',
            output_path
        ])

        cmd_string = ' '.join(shlex.quote(str(arg)) for arg in ffmpeg_command)
        print(f"Executing FFmpeg: {cmd_string}")

        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8')
        
        print(f"Successfully created vertical short at: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg execution:")
        print(f"  Command: {' '.join(e.cmd)}")
        print(f"  Return Code: {e.returncode}")
        print(f"  Stdout: {e.stdout.strip()}")
        print(f"  Stderr: {e.stderr.strip()}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred in process_frame_for_vertical_short: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Очистка временных файлов
        if ass_file_path and os.path.exists(ass_file_path):
            try:
                os.remove(ass_file_path)
            except OSError as e:
                print(f"Warning: could not remove temp ass file: {e}")
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except OSError as e:
                print(f"Warning: could not remove temp directory: {e}")
