from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip
import subprocess
import math
import tempfile
import os
import shlex
import json # For ffprobe output

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
        return None, None

def create_shorts_with_blur_background(input_video_path, output_video_path, target_ratio_width=213, target_ratio_height=274):
    """
    Создает shorts с размытым фоновым слоем согласно спецификации:
    - Центральное видео с правильными пропорциями для целевого соотношения сторон
    - Размытый фоновый слой, заполняющий всю область финального видео
    - Соотношение сторон target_ratio_width:target_ratio_height (по умолчанию 213:274)
    - Полосы сверху и снизу корректно заполнены размытым фоном
    - Сохранение оригинального аудио
    - Высокое качество MP4

    Args:
        input_video_path (str): Путь к входному видео
        output_video_path (str): Путь для сохранения результата
        target_ratio_width (int): Ширина целевого соотношения сторон
        target_ratio_height (int): Высота целевого соотношения сторон

    Returns:
        bool: True если успешно, False при ошибке
    """
    try:
        print(f"Начинаю обработку видео для shorts: {input_video_path}")

        # 1. Получаем размеры оригинального видео
        original_width, original_height = get_video_dimensions(input_video_path)
        if original_width is None or original_height is None:
            print("Ошибка: Не удалось получить размеры видео")
            return False

        print(f"Оригинальные размеры: {original_width}x{original_height}")
        print(f"Оригинальное соотношение сторон: {original_width/original_height:.3f}")

        # 2. Рассчитываем размеры финального видео для соотношения target_ratio_width:target_ratio_height
        target_ratio = target_ratio_width / target_ratio_height
        print(f"Целевое соотношение сторон: {target_ratio:.3f}")

        # Определяем размеры финального видео на основе оригинального видео
        original_ratio = original_width / original_height

        if original_ratio > target_ratio:
            # Оригинальное видео шире целевого - ограничиваем по высоте
            final_height = original_height
            final_width = int(final_height * target_ratio)
        else:
            # Оригинальное видео выше целевого - ограничиваем по ширине
            final_width = original_width
            final_height = int(final_width / target_ratio)

        # Обеспечиваем четные размеры для совместимости с кодеками
        final_width = final_width if final_width % 2 == 0 else final_width - 1
        final_height = final_height if final_height % 2 == 0 else final_height - 1

        print(f"Целевые размеры финального видео: {final_width}x{final_height}")
        print(f"Финальное соотношение сторон: {final_width/final_height:.3f}")

        # 3. Рассчитываем размеры основного видео (центральная часть для наложения)
        # Основное видео должно занимать центральную область финального видео с сохранением пропорций
        main_video_ratio = original_width / original_height

        if final_width / final_height > main_video_ratio:
            # Финальное видео шире - ограничиваем основное видео по высоте
            main_video_height = final_height
            main_video_width = int(main_video_height * main_video_ratio)
        else:
            # Финальное видео выше - ограничиваем основное видео по ширине
            main_video_width = final_width
            main_video_height = int(main_video_width / main_video_ratio)

        # Центрируем основное видео в финальном кадре
        main_x_offset = (final_width - main_video_width) // 2
        main_y_offset = (final_height - main_video_height) // 2

        print(f"Размеры основного видео: {main_video_width}x{main_video_height}")
        print(f"Позиция основного видео: ({main_x_offset}, {main_y_offset})")
        print(f"Соотношение сторон основного видео: {main_video_width/main_video_height:.3f}")

        # 4. Создаем временные файлы
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            # Пути для временных файлов
            main_video_temp = os.path.join(temp_dir, "main_video.mp4")
            blur_background_temp = os.path.join(temp_dir, "blur_background.mp4")
            final_composition_temp = os.path.join(temp_dir, "final_composition.mp4")

            # 5. Создаем основное видео с правильными пропорциями
            # Вырезаем центральную часть оригинального видео для основного контента
            crop_width = min(original_width, int(original_height * main_video_ratio))
            crop_height = min(original_height, int(original_width / main_video_ratio))
            crop_x = (original_width - crop_width) // 2
            crop_y = (original_height - crop_height) // 2

            main_video_command = [
                'ffmpeg',
                '-i', input_video_path,
                '-vf', f'crop={crop_width}:{crop_height}:{crop_x}:{crop_y}',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',  # Высокое качество
                '-c:a', 'copy',
                '-y',
                main_video_temp
            ]

            print("Создаю основное видео...")
            process = subprocess.run(main_video_command, check=True, capture_output=True, text=True)
            print(f"Основное видео создано: {main_video_temp}")

            # Проверяем размеры созданного основного видео
            main_temp_width, main_temp_height = get_video_dimensions(main_video_temp)
            print(f"Размеры основного видео: {main_temp_width}x{main_temp_height}")

            # 6. Создаем размытый фоновый слой, который заполнит всю область финального видео
            # Используем оригинальное видео как источник для фона, масштабируя его под финальные размеры
            blur_strength = max(5, min(15, original_width // 200))  # От 5 до 15 в зависимости от ширины
            print(f"Сила размытия: {blur_strength} (адаптивно для разрешения {original_width}x{original_height})")
    
            # Debug: Check blur background parameters
            print(f"Blur background target size: {final_width}x{final_height}")
            print(f"Blur strength validation: {blur_strength} (recommended range: 5-15)")
            if blur_strength < 5 or blur_strength > 15:
                print(f"WARNING: Blur strength {blur_strength} outside recommended range!")
            else:
                print("INFO: Blur strength within recommended range")

            blur_background_command = [
                'ffmpeg',
                '-i', input_video_path,
                '-vf', f'scale={final_width}:{final_height}:force_original_aspect_ratio=decrease,pad={final_width}:{final_height}:(ow-iw)/2:(oh-ih)/2,boxblur={blur_strength}:{blur_strength}',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '28',  # Более низкое качество для фона
                '-c:a', 'copy',
                '-y',
                blur_background_temp
            ]

            print(f"Создаю размытый фоновый слой {final_width}x{final_height}...")
            process = subprocess.run(blur_background_command, check=True, capture_output=True, text=True)
            print(f"Размытый фон создан: {blur_background_temp}")

            # Проверяем размеры созданного размытого фона
            blur_temp_width, blur_temp_height = get_video_dimensions(blur_background_temp)
            print(f"Размеры размытого фона: {blur_temp_width}x{blur_temp_height}")

            # 7. Компонуем финальное видео с наложением слоев
            print(f"Размеры наложения основного видео: {main_video_width}x{main_video_height}")
            print(f"Позиция наложения: ({main_x_offset}, {main_y_offset})")

            # Проверяем, что размеры наложения корректны
            if main_video_width <= 0 or main_video_height <= 0:
                print("Ошибка: Некорректные размеры наложения основного видео")
                return False

            composition_command = [
                'ffmpeg',
                '-i', blur_background_temp,  # Фоновый слой
                '-i', main_video_temp,       # Основной слой
                '-filter_complex',
                f'[0:v][1:v]overlay={main_x_offset}:{main_y_offset}',
                '-c:v', 'libx264',
                '-preset', 'slow',  # Более высокое качество для финального видео
                '-crf', '18',       # Высокое качество
                '-c:a', 'copy',     # Сохраняем оригинальное аудио
                '-y',
                final_composition_temp
            ]

            print("Компонуем финальное видео...")
            process = subprocess.run(composition_command, check=True, capture_output=True, text=True)
            print(f"Композиция создана: {final_composition_temp}")

            # Проверяем размеры финального видео
            final_temp_width, final_temp_height = get_video_dimensions(final_composition_temp)
            print(f"Размеры финального видео: {final_temp_width}x{final_temp_height}")

            if final_temp_width != final_width or final_temp_height != final_height:
                print(f"⚠️  Предупреждение: Размеры финального видео не совпадают. Ожидалось {final_width}x{final_height}, получено {final_temp_width}x{final_temp_height}")

            # 8. Финальная обработка - копируем в выходной файл
            final_command = [
                'ffmpeg',
                '-i', final_composition_temp,
                '-c', 'copy',  # Простое копирование без перекодирования
                '-y',
                output_video_path
            ]

            print(f"Сохраняю финальное видео: {output_video_path}")
            process = subprocess.run(final_command, check=True, capture_output=True, text=True)

            # Финальная проверка выходного файла
            output_width, output_height = get_video_dimensions(output_video_path)
            if output_width and output_height:
                print(f"✅ Shorts успешно создан: {output_video_path}")
                print(f"✅ Финальные размеры: {output_width}x{output_height}")
                print(f"✅ Соотношение сторон: {output_width/output_height:.3f} (целевое: {target_ratio:.3f})")

                # Debug: Final validation
                actual_ratio = output_width / output_height
                ratio_diff = abs(actual_ratio - target_ratio)
                print(f"Final aspect ratio difference: {ratio_diff:.3f}")
                if ratio_diff > 0.01:
                    print(f"WARNING: Final aspect ratio deviates from target by {ratio_diff:.3f}")
                else:
                    print("INFO: Final aspect ratio matches target")

                # Check if blur background is properly applied (basic check)
                print("INFO: Blur background should be present in final composition")
                return True
            else:
                print("❌ Ошибка: Не удалось проверить финальный файл")
                return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка FFmpeg: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка при создании shorts: {e}")
        import traceback
        traceback.print_exc()
        return False


# Example usage:
# if __name__ == "__main__":
#    # ... (old example usage)

