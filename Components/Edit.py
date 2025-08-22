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
import torch

# Import caption functions from the new module
from .Captions import create_ass_file
from .FaceCrop import crop_to_vertical_dynamic_smoothed, analyze_face_position_lightweight # Import the new function

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
    Creates a vertical short by first cropping a segment, then applying dynamic face tracking,
    and finally adding animated captions.
    """
    temp_dir = None
    ass_file_path = None
    temp_segment_path = None
    temp_cropped_path = None

    try:
        # 1. --- Setup and Validation ---
        if not os.path.exists(source_video_path):
            print(f"Error: Source video not found at '{source_video_path}'")
            return False
        if end_time <= start_time:
            print("Error: End time must be greater than start time.")
            return False

        temp_dir = tempfile.mkdtemp()
        temp_segment_path = os.path.join(temp_dir, "segment.mp4")
        temp_cropped_path = os.path.join(temp_dir, "cropped_video.mp4")

        # 2. --- Extract Video Segment (Fast) ---
        print(f"Extracting segment from {start_time:.2f}s to {end_time:.2f}s...")
        if not crop_video(source_video_path, temp_segment_path, start_time, end_time, 0, 0):
             print("Error: Failed to extract video segment.")
             return False

        # 3. --- Apply Dynamic Smooth Crop ---
        print("Applying dynamic smooth face crop...")
        cropped_video_path = crop_to_vertical_dynamic_smoothed(
            input_video_path=temp_segment_path,
            output_video_path=temp_cropped_path,
            face_cascade_path=face_cascade_path
        )
        if not cropped_video_path:
            print("Error: Dynamic face cropping failed.")
            return False

        # 4. --- Get Dimensions of the Final Cropped Video ---
        final_width, final_height = get_video_dimensions(cropped_video_path)
        if not final_width or not final_height:
             print("Error: Could not get dimensions of the final cropped video.")
             return False

        # 5. --- Create Animated Captions ---
        print("Preparing animated captions...")
        ass_file_path = os.path.join(temp_dir, "captions.ass")
        ass_success = create_ass_file(
            word_level_transcription,
            ass_file_path,
            final_width,
            final_height,
            segment_start_time=0, # Timestamps in ASS are relative to the segment
            segment_end_time=(end_time - start_time)
        )
        if not ass_success:
            print("Warning: Failed to create ASS subtitle file. Proceeding without captions.")
            ass_file_path = None

        # 6. --- Burn Captions into Video ---
        print("Building and running final FFmpeg command to burn captions...")
        ffmpeg_command = [
            'ffmpeg',
            '-i', cropped_video_path,
            '-c:a', 'copy',
        ]

        # Add ASS filter only if the file was created successfully
        if ass_file_path and os.path.exists(ass_file_path):
             escaped_ass_path = ass_file_path.replace('\\', '/').replace(':', '\\\\:')
             ffmpeg_command.extend(['-vf', f"ass='{escaped_ass_path}'"])

        ffmpeg_command.extend([
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-aspect', '9:16',
            '-y',
            output_path
        ])

        cmd_string = ' '.join(shlex.quote(str(arg)) for arg in ffmpeg_command)
        print(f"Executing FFmpeg: {cmd_string}")
        
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8')
        
        print(f"Successfully created vertical short at: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg execution:")
        print(f"  Command: {' '.join(e.cmd)}")
        print(f"  Return Code: {e.returncode}")
        print(f"  Stdout: {e.stdout.strip() if e.stdout else 'N/A'}")
        print(f"  Stderr: {e.stderr.strip() if e.stderr else 'N/A'}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred in process_frame_for_vertical_short: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # --- Cleanup ---
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except OSError as e:
                print(f"Warning: could not remove temp directory: {e}")


def process_highlight_unified(
    source_video_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    word_level_transcription: dict,
    crop_bottom_percent: float = 0.0,
    face_cascade_path: str = 'haarcascade_frontalface_default.xml'
) -> bool:
    """
    Unified processing function that creates a vertical short using a single FFmpeg pipeline.
    This eliminates multiple read/write operations by combining crop and ass filters.
    """
    temp_dir = None
    ass_file_path = None

    try:
        # 1. --- Setup and Validation ---
        if not os.path.exists(source_video_path):
            print(f"Error: Source video not found at '{source_video_path}'")
            return False
        if end_time <= start_time:
            print("Error: End time must be greater than start time.")
            return False

        temp_dir = tempfile.mkdtemp()
        ass_file_path = os.path.join(temp_dir, "captions.ass")

        # 2. --- Analyze Face Position ---
        print("Analyzing face position for crop parameters...")
        avg_face_center_x = analyze_face_position_lightweight(
            video_path=source_video_path,
            sample_rate=25
        )

        if avg_face_center_x is None or avg_face_center_x == 0.0:
            print("Warning: Could not analyze face position, using center crop")
            avg_face_center_x = original_width / 2  # Default to center in pixels
        else:
            # Convert to normalized value (0.0 to 1.0)
            avg_face_center_x = avg_face_center_x / original_width

        # 3. --- Calculate Crop Parameters ---
        # For vertical video (9:16 aspect ratio)
        target_aspect = 9/16
        original_width, original_height = get_video_dimensions(source_video_path)

        if not original_width or not original_height:
            print("Error: Could not get video dimensions")
            return False

        # Calculate crop dimensions
        crop_height = original_height
        crop_width = int(crop_height * target_aspect)

        # Ensure crop width doesn't exceed original width
        if crop_width > original_width:
            crop_width = original_width
            crop_height = int(crop_width / target_aspect)

        # Calculate crop x position based on face center (avg_face_center_x is normalized 0.0-1.0)
        crop_x = int((original_width - crop_width) * avg_face_center_x)
        # Ensure crop doesn't go outside video bounds
        crop_x = max(0, min(crop_x, original_width - crop_width))

        # Apply bottom crop percentage if specified
        if crop_bottom_percent > 0:
            crop_height = int(crop_height * (1.0 - crop_bottom_percent))
            # Ensure even height for FFmpeg
            crop_height = crop_height - (crop_height % 2)

        crop_filter = f"crop={crop_width}:{crop_height}:{crop_x}:0"

        # 4. --- Create ASS Subtitle File ---
        print("Creating ASS subtitle file...")
        ass_success = create_ass_file(
            word_level_transcription,
            ass_file_path,
            crop_width,  # Use cropped dimensions
            crop_height,
            segment_start_time=0,
            segment_end_time=(end_time - start_time)
        )

        if not ass_success:
            print("Warning: Failed to create ASS subtitle file. Proceeding without captions.")
            ass_file_path = None

        # 5. --- Check GPU Availability and Build Single FFmpeg Command ---
        print("Checking GPU availability for hardware acceleration...")
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            print("NVIDIA GPU detected - using GPU acceleration (NVENC)")
            video_codec = 'h264_nvenc'
            hwaccel_flag = ['-hwaccel', 'cuda']
        else:
            print("No NVIDIA GPU detected - using CPU encoding (libx264)")
            video_codec = 'libx264'
            hwaccel_flag = []

        print("Building unified FFmpeg command with crop and ass filters...")

        ffmpeg_command = [
            'ffmpeg'
        ] + hwaccel_flag + [
            '-ss', str(start_time),
            '-i', source_video_path,
            '-t', str(end_time - start_time),
            '-c:a', 'copy'
        ]

        # Build video filter chain
        video_filters = [crop_filter]

        # Add ASS filter if subtitle file was created
        if ass_file_path and os.path.exists(ass_file_path):
            escaped_ass_path = ass_file_path.replace('\\', '/').replace(':', '\\\\:')
            video_filters.append(f"ass='{escaped_ass_path}'")

        # Join filters with comma
        if video_filters:
            ffmpeg_command.extend(['-vf', ','.join(video_filters)])

        # Output settings
        ffmpeg_command.extend([
            '-c:v', video_codec,
            '-preset', 'medium',
            '-crf', '23',
            '-aspect', '9:16',
            '-y',
            output_path
        ])

        # 6. --- Execute FFmpeg Command ---
        cmd_string = ' '.join(shlex.quote(str(arg)) for arg in ffmpeg_command)
        print(f"Executing unified FFmpeg command: {cmd_string}")

        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"Successfully created unified vertical short at: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during unified FFmpeg execution:")
        print(f"  Command: {' '.join(e.cmd)}")
        print(f"  Return Code: {e.returncode}")
        print(f"  Stdout: {e.stdout.strip() if e.stdout else 'N/A'}")
        print(f"  Stderr: {e.stderr.strip() if e.stderr else 'N/A'}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred in process_highlight_unified: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # --- Cleanup ---
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except OSError as e:
                print(f"Warning: could not remove temp directory: {e}")
