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
    try:
        # --- Стили ---
        # Рассчитываем размер шрифта относительно высоты видео
        font_size = int(video_height / 25)
        outline = int(font_size / 10)
        shadow = int(outline / 2)
        margin_v = int(video_height / 15)

        ass_header = f"""[Script Info]
Title: AI Generated Captions
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat, {font_size},&H00FFFFFF,&H000000FF,&H00000000,&H60000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,10,10,{margin_v},1

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

# Function removed: burn_captions was deprecated and unused


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

def animate_captions(vertical_video_path, audio_source_path, transcription_result, output_path):
    """Creates a video with word-by-word highlighted captions based on segments."""
    temp_animated_video = output_path + "_temp_anim.mp4"
    success = False
    cap = None
    out = None

    try:
        # --- Font Setup (Pillow) ---
        font_size = 34  # Adjust size as needed
        base_dir = os.path.dirname(os.path.abspath(__file__))
        abs_font_path = os.path.normpath(os.path.join(base_dir, "..", "fonts", "Montserrat-Bold.ttf"))
        font = None
        try:
            if os.path.exists(abs_font_path):
                font = ImageFont.truetype(abs_font_path, font_size)
                print(f"Successfully loaded font: {abs_font_path}")
            else:
                print(f"Font file not found at {abs_font_path}. Using PIL default font.")
                font = ImageFont.load_default()
        except Exception as e:
            print(f"Warning: could not load TTF font at {abs_font_path}: {e}. Using PIL default font.")
            font = ImageFont.load_default()
        # --- Pre-filter segments ---
        original_segments = transcription_result.get("segments", [])
        filtered_segments = [seg for seg in original_segments if seg.get('text', '').strip() != '[*]']
        if not filtered_segments:
            print("Warning: No non-[*] segments found in transcription. Captions might be empty.")
            # Optional: Decide if you want to proceed with an empty list or return early
            # return False # Example: Exit if no valid captions

        # Update the transcription_result to use filtered segments for further processing
        transcription_result_filtered = transcription_result.copy() # Avoid modifying original dict directly if reused
        transcription_result_filtered['segments'] = filtered_segments

        print("Starting animated caption generation (Static Window/Highlight Style)...")
        cap = cv2.VideoCapture(vertical_video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {vertical_video_path}")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: # Handle zero or negative fps
             print(f"Error: Invalid video FPS ({fps}), cannot calculate time.")
             cap.release() # Release resource
             return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Input video properties: {width}x{height} @ {fps:.2f}fps")

        # --- Text Styling (Pillow) ---
        # Use RGB for Pillow colors
        text_color_rgb = (255, 255, 0)  # Yellow
        stroke_color_rgb = (0, 0, 0)    # Black outline
        stroke_width = 1                # Outline width in pixels
        bottom_margin = 120 # Increased margin (moves text higher)

        # --- Calculate Fixed Y Position (after getting height) ---
        font_ascent = 0 # Default if font fails
        if font:
            try:
                font_ascent, _ = font.getmetrics()
            except AttributeError: # Handle cases where getmetrics might not exist? (Shouldn't for TTF)
                 print("Warning: Could not get font metrics.")
        fixed_top_y = height - bottom_margin - font_ascent

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_animated_video, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Error: Could not open video writer for {temp_animated_video}")
            cap.release() # Release resource
            return False

        frame_count = 0
        drawn_any_text = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / fps

            # Find the segment and specific word active at this frame's time
            # Use the filtered transcription data
            active_segment, active_word_idx_in_segment = find_active_segment_and_word(transcription_result_filtered, current_time)

            # --- Get Words to Display (Max 2) ---
            words_to_display = ""
            if active_segment:
                segment_words_list = active_segment.get('words', [])
                num_words_in_segment = len(segment_words_list)

                if 0 <= active_word_idx_in_segment < num_words_in_segment:
                    word1_info = segment_words_list[active_word_idx_in_segment]
                    word1_text = word1_info.get('text', '').strip()

                    # Skip if the primary word is [*]
                    if word1_text != '[*]':
                        words_to_display = word1_text
                        # Try to get the next word
                        next_word_idx = active_word_idx_in_segment + 1
                        if next_word_idx < num_words_in_segment:
                            word2_info = segment_words_list[next_word_idx]
                            word2_text = word2_info.get('text', '').strip()
                            # Also skip if the second word is [*]
                            if word2_text != '[*]':
                                words_to_display += f" {word2_text}"

            # --- Drawing Logic (Pillow - Max 2 words) ---
            # Draw only if we have a valid window and an active word within it
            if words_to_display and font: # Only draw if we have words and font loaded
                # Convert frame BGR OpenCV to RGB Pillow
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(pil_image)

                # Get text dimensions
                text_bbox = draw.textbbox((0, 0), words_to_display, font=font)
                text_width = text_bbox[2] - text_bbox[0]

                # Calculate position (centered horizontally)
                start_x = (width - text_width) // 2

                # Draw the text (single color)
                draw.text(
                    (start_x, fixed_top_y), # Use fixed vertical position
                    words_to_display,
                    font=font,
                    fill=text_color_rgb,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_color_rgb
                )
                drawn_any_text = True

                # Convert back to OpenCV BGR format
                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # --- Write frame ---
            out.write(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                 print(f"Processed {frame_count}/{total_frames} frames for animation...")

        print("Finished processing frames for animation.")

    except Exception as e:
        print(f"Error during caption animation loop: {e}")
        traceback.print_exc() # Print detailed traceback
        success = False
    finally:
        print("Releasing video resources...")
        if cap and cap.isOpened():
            cap.release()
        if out and out.isOpened():
            out.release()

        # Proceed with muxing only if frames were processed, some text was drawn, and temp file exists
        if drawn_any_text and frame_count > 0 and os.path.exists(temp_animated_video):
             try:
                 print("Muxing audio into animated video...")
                 ffmpeg_mux_command = [
                     'ffmpeg',
                     '-i', temp_animated_video,
                     '-i', audio_source_path,
                     '-map', '0:v:0',
                     '-map', '1:a:0',
                     '-c:v', 'copy',
                     '-c:a', 'aac',
                     '-b:a', '128k',
                     '-shortest',
                     '-y',
                     output_path
                 ]
                 cmd_string = ' '.join([str(arg) for arg in ffmpeg_mux_command])
                 print(f"Mux Command: {cmd_string}")
                 process = subprocess.run(ffmpeg_mux_command, check=True, capture_output=True, text=True, timeout=300)
                 print(f"Successfully created animated caption video: {output_path}")
                 success = True
             except subprocess.TimeoutExpired:
                 print("Error: FFmpeg muxing timed out.")
                 success = False
             except subprocess.CalledProcessError as mux_e:
                  print(f"Error during audio muxing (FFmpeg): {mux_e}")
                  print(f"FFmpeg stdout: {mux_e.stdout}")
                  print(f"FFmpeg stderr: {mux_e.stderr}")
                  success = False
             except Exception as mux_e:
                  print(f"An unexpected error occurred during audio muxing: {mux_e}")
                  success = False
             finally:
                 # Ensure cleanup even if muxing fails
                 if os.path.exists(temp_animated_video):
                     try:
                         os.remove(temp_animated_video)
                         print(f"Removed temporary animated video: {temp_animated_video}")
                     except Exception as e_clean:
                         print(f"Warning: Could not remove temp animated file: {e_clean}")
        elif frame_count > 0 and os.path.exists(temp_animated_video):
             print("No text was drawn on any frame; skipping audio muxing for animated captions.")
             success = False
             # Cleanup temp animated video file
             try:
                 os.remove(temp_animated_video)
                 print(f"Removed temporary animated video: {temp_animated_video}")
             except Exception as e_clean:
                 print(f"Warning: Could not remove temp animated file: {e_clean}")
        elif not os.path.exists(temp_animated_video) and frame_count > 0:
             print(f"Error: Temp animated video file {temp_animated_video} not found, cannot mux audio.")
             success = False
        else: # frame_count == 0 or initial error before loop
             print("Skipping audio muxing due to processing error or no frames processed.")
             success = False # Ensure success is false if animation failed early

    return success 