from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip
import subprocess
import math
import tempfile
import os
import shlex
import json # For ffprobe output
import torch

import cv2
import numpy as np

# Import caption functions from the new module
from .Captions import burn_captions, animate_captions, create_ass_file
from .FaceCrop import analyze_face_position_lightweight # Import the new function
import logging

from .config import get_config

def process_highlight_unified(source_video: str, start_time: float, end_time: float, transcript_data: dict, output_path: str):
    """
    Processes a video highlight using a unified FFmpeg command for cropping and adding subtitles.
    """
    cfg = get_config()
    logging.info(f"Starting unified processing for {source_video} from {start_time} to {end_time}")

    try:
        # 1. Pre-analysis to find the average face position
        logging.info("Analyzing face position...")
        avg_face_center_x = analyze_face_position_lightweight(source_video, cfg.video_processing.face_detection_sample_rate)
        if avg_face_center_x == 0:
            logging.warning("Could not detect face, using center of the frame as fallback.")
            # Get video width to fall back to center
            cap = cv2.VideoCapture(source_video)
            if not cap.isOpened():
                logging.error("Failed to open source video with OpenCV to get width.")
                return False
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            avg_face_center_x = video_width / 2
        
        logging.info(f"Average face center X: {avg_face_center_x}")

        # 2. Calculate cropping geometry
        crop_h = cfg.video_processing.crop_height
        crop_w = cfg.video_processing.crop_width
        
        crop_x = int(avg_face_center_x - crop_w / 2)

        # Get original video width for boundary checks
        cap = cv2.VideoCapture(source_video)
        if not cap.isOpened():
            logging.error("Failed to open source video with OpenCV for dimensions.")
            return False
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        # Clamp crop_x to be within frame boundaries
        crop_x = max(0, crop_x)
        crop_x = min(crop_x, original_width - crop_w)
        
        logging.info(f"Calculated crop geometry: w={crop_w}, h={crop_h}, x={crop_x}")

        # 3. Generate subtitles
        temp_dir = tempfile.mkdtemp()
        ass_path = os.path.join(temp_dir, "captions.ass")
        logging.info(f"Generating subtitles at: {ass_path}")

        create_ass_file(
            word_level_transcription=transcript_data,
            output_ass_path=ass_path,
            video_width=crop_w,
            video_height=crop_h,
            segment_start_time=start_time,
            segment_end_time=end_time
        )
        
        # FFmpeg requires escaping special characters in paths
        escaped_ass_path = ass_path.replace('\\', '/').replace(':', '\\:')

        # 4. Form and execute FFmpeg command
        duration = end_time - start_time
        
        # Dynamically build the FFmpeg command based on GPU availability
        gpu_available = torch.cuda.is_available()
        logging.info(f"NVIDIA GPU available: {gpu_available}")

        if gpu_available:
            # Use NVIDIA's NVENC for hardware-accelerated encoding
            video_codec_options = f"-hwaccel cuda -c:v h264_nvenc -preset {cfg.ffmpeg.gpu_preset}"
            logging.info("Using GPU-accelerated (h264_nvenc) FFmpeg command.")
        else:
            # Fallback to CPU-based encoding
            video_codec_options = f"-c:v {cfg.ffmpeg.cpu_codec} -preset {cfg.ffmpeg.cpu_preset}"
            logging.info(f"Using CPU-based ({cfg.ffmpeg.cpu_codec}) FFmpeg command.")

        ffmpeg_command = (
            f'ffmpeg -y -ss {start_time} -i "{source_video}" -t {duration} '
            f'-vf "crop={crop_w}:{crop_h}:{crop_x}:0,ass=\'{escaped_ass_path}\'" '
            f'-c:a copy {video_codec_options} "{output_path}"'
        )

        logging.info(f"Executing FFmpeg command:\n{ffmpeg_command}")

        # 5. Execution and cleanup
        process = subprocess.run(ffmpeg_command, shell=True, check=True, capture_output=True, text=True)
        
        logging.info("FFmpeg execution successful.")
        logging.info(f"Output video saved to: {output_path}")

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg command failed with exit code {e.returncode}")
        logging.error(f"FFmpeg stderr:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up the temporary .ass file
        if 'ass_path' in locals() and os.path.exists(ass_path):
            try:
                os.remove(ass_path)
                logging.info(f"Removed temporary subtitles file: {ass_path}")
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {ass_path}: {e}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logging.info(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                logging.warning(f"Failed to remove temporary directory {temp_dir}: {e}")

    return True

def extractAudio(video_path):
    cfg = get_config()
    try:
        video_clip = VideoFileClip(video_path)
        audio_path = cfg.video_processing.temp_audio_filename
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


