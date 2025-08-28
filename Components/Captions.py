import math
import string
import cv2
import numpy as np # Add numpy import
import os
import subprocess
import traceback # For detailed error printing in animate_captions
from PIL import Image, ImageDraw, ImageFont # Pillow imports for custom font
from Components.Paths import fonts_path

# pure helper for unit tests and reuse
def _compute_bottom_margin_px(frame_h: int, bottom_offset_pct: int) -> int:
    try:
        # positioning via bottom_offset_pct
        pct = int(bottom_offset_pct)
    except Exception:
        pct = 0
    pct = max(0, min(pct, 100))
    try:
        fh = int(frame_h)
    except Exception:
        fh = 0
    return int(fh * pct / 100)
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

# Function to generate ASS content (more compatible with ffmpeg filter)
def generate_ass_content(transcriptions, start_time, end_time, style_cfg=None, video_w=None, video_h=None):
    """
    Генерирует ASS-контент. Обратная совместимость сохранена:
    - При style_cfg is None — используются прежние константы (PlayRes 384x720, Fontsize=36, цвета/отступы как раньше).
    - При наличии style_cfg — применяются параметры из конфигурации:
      font_size, base_color, shadow (offset -> Shadow, color -> BackColour), letter_spacing -> Spacing,
      позиционирование Alignment/MarginV через position.mode и bottom_offset_pct.
    """

    # Локальный helper: HEX (#RRGGBB или #RRGGBBAA) -> &HAABBGGRR
    def _hex_to_ass_color(hex_str: str, default: str = "#FFFFFFFF") -> str:
        try:
            s = str(hex_str).strip()
            if not s.startswith("#"):
                raise ValueError("No #")
            s = s[1:]
            if len(s) == 6:
                rr, gg, bb = s[0:2], s[2:4], s[4:6]
                aa = "00"  # 00 — непрозрачно в ASS
            elif len(s) == 8:
                rr, gg, bb, aa = s[0:2], s[2:4], s[4:6], s[6:8]
            else:
                raise ValueError("Bad length")
            # Формат &HAABBGGRR
            return f"&H{aa}{bb}{gg}{rr}"
        except Exception:
            # На некорректный ввод — вернуть default
            if default != hex_str:
                return _hex_to_ass_color(default, "#FFFFFFFF")
            # Подстраховка (белый непрозрачный)
            return "&H00FFFFFF"

    # Дефолтные значения (как в предыдущей реализации)
    use_style = style_cfg is not None
    default_play_x, default_play_y = 384, 720
    play_res_x = default_play_x
    play_res_y = default_play_y
    if use_style and isinstance(video_w, (int, float)) and isinstance(video_h, (int, float)) and video_w > 0 and video_h > 0:
        # Используем реальный размер видео только если style_cfg задан (чтобы не менять старое поведение)
        play_res_x = int(video_w)
        play_res_y = int(video_h)

    font_size_default = 38
    font_size = font_size_default
    primary_colour = "&H00FFFFFF"  # белый
    outline_colour = "&H00000000"  # чёрный
    back_colour = "&H70000000"     # как было в коде
    outline = 2
    shadow_val = 1
    spacing_val = 0
    alignment = 2  # bottom-center
    margin_l = 10
    margin_r = 10
    margin_v = 30  # как было

    if use_style:
        # Font size
        try:
            fs = int(getattr(style_cfg, "font_size_px", font_size_default))
            font_size = max(20, min(60, fs))
        except Exception:
            font_size = font_size_default

        # Primary colour (base_color)
        try:
            primary_colour = _hex_to_ass_color(getattr(style_cfg, "base_color", "#FFFFFF"))
        except Exception:
            primary_colour = "&H00FFFFFF"

        # Back colour from shadow.color
        try:
            shadow_cfg = getattr(style_cfg, "shadow", None)
            if shadow_cfg:
                back_colour = _hex_to_ass_color(getattr(shadow_cfg, "color", "#00000080"))
                # Shadow offset (ASS поддерживает только интенсивность)
                sx = int(getattr(shadow_cfg, "x_px", 2) or 0)
                sy = int(getattr(shadow_cfg, "y_px", 2) or 0)
                shadow_int = max(sx, sy)
                try:
                    shadow_val = max(1, min(4, int(round(shadow_int))))
                except Exception:
                    shadow_val = 1
        except Exception:
            pass

        # Letter spacing
        try:
            spacing_val = int(round(float(getattr(style_cfg, "letter_spacing_px", 0) or 0)))
        except Exception:
            spacing_val = 0

        # Alignment/MarginV
        # positioning via bottom_offset_pct and center_offset_pct
        # Alignment mapping: safe_bottom→2, center→5
        try:
            position = getattr(style_cfg, "position", None)
            mode = getattr(position, "mode", "safe_bottom") if position else "safe_bottom"
            bottom_offset_pct = int(getattr(position, "bottom_offset_pct", 22)) if position else 22
            center_offset_pct = int(getattr(position, "center_offset_pct", 12)) if position else 12
            if mode == "center":
                alignment = 5  # middle-center
                # Для режима center рассчитываем margin_v как смещение ниже центра
                try:
                    center_y = play_res_y // 2
                    offset_px = int(play_res_y * center_offset_pct / 100.0)
                    margin_v = play_res_y - (center_y + offset_px)  # Расстояние от низа до позиции текста
                except Exception:
                    margin_v = play_res_y // 2 - 50  # Резервное значение
            else:
                alignment = 2  # bottom-center
                try:
                    margin_v = int(play_res_y * float(bottom_offset_pct) / 100.0)
                except Exception:
                    margin_v = 30
        except Exception:
            pass

    ass_content = (
        f"[Script Info]\n"
        f"Title: Auto-generated by AI-Youtube-Shorts-Generator\n"
        f"ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        f"WrapStyle: 0\n"
        f"ScaledBorderAndShadow: yes\n"
        f"\n"
        f"[V4+ Styles]\n"
        f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        f"OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        f"ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        f"MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Poppins,{font_size},{primary_colour},&H000000FF,{outline_colour},{back_colour},"
        f"-1,0,0,0,100,100,{spacing_val},0,1,{outline},{shadow_val},{alignment},{margin_l},{margin_r},{margin_v},1\n"
        f"Style: Fallback,Arial,{font_size},{primary_colour},&H000000FF,{outline_colour},{back_colour},"
        f"-1,0,0,0,100,100,{spacing_val},0,1,{outline},{shadow_val},{alignment},{margin_l},{margin_r},{margin_v},1\n"
        f"\n"
        f"[Events]\n"
        f"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    # Итерация по сегментам и словам для создания "караоке" эффекта (появление слов по одному)
    segments = transcriptions.get("segments", [])
    for segment in segments:
        # Пропускаем сегменты без слов
        if not segment.get("words"):
            continue

        for word_info in segment.get("words", []):
            # В новой структуре текст слова в 'word', но для совместимости проверим и 'text'
            word_text = (word_info.get('word') or word_info.get('text', '')).strip()

            # Пропускаем пустые слова или специальные маркеры
            if not word_text or word_text == '[*]':
                continue

            word_start = word_info.get('start')
            word_end = word_info.get('end')

            # Пропускаем слова без временных меток
            if word_start is None or word_end is None:
                continue

            # Проверяем, что слово попадает в указанный временной диапазон
            if word_start >= start_time and word_end <= end_time:
                relative_start = word_start - start_time
                relative_end = word_end - start_time
                
                if relative_start < 0:
                    relative_start = 0.0
                # Убедимся, что длительность положительная
                if relative_end <= relative_start:
                    relative_end = relative_start + 0.1

                # Применяем очистку от пунктуации и перевод в верхний регистр к каждому слову
                clean_text = word_text.upper().translate(str.maketrans('', '', string.punctuation))

                # Генерируем событие 'Dialogue' для каждого слова
                ass_content += (
                    f"Dialogue: 0,{format_time_ass(relative_start)},{format_time_ass(relative_end)},"
                    f"Default,,0,0,0,,{clean_text}\\N"
                )

    return ass_content

# Function to burn captions using FFmpeg
def burn_captions(vertical_video_path, audio_source_path, transcriptions, start_time, end_time, output_path, style_cfg=None):
    """Burns captions onto the vertical video using audio from the source segment."""
    temp_ass_path = "temp_subtitles.ass"  # Simple name in current directory
    # Пытаемся получить реальные размеры видео (для PlayRes при наличии style_cfg)
    vw, vh = None, None
    try:
        cap = cv2.VideoCapture(vertical_video_path)
        if cap.isOpened():
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if cap:
            cap.release()
    except Exception:
        vw, vh = None, None

    try:
        # Create an ASS subtitle file (more compatible than SRT for styling)
        ass_content = generate_ass_content(transcriptions, start_time, end_time, style_cfg=style_cfg, video_w=vw, video_h=vh)

        if not ass_content.count("Dialogue:"):
            print("No relevant transcriptions found for the highlight duration. Using video without captions.")
            # Need to add audio even if no captions are burned
            ffmpeg_command_no_subs = [
                'ffmpeg',
                '-i', vertical_video_path, # Silent video input
                '-i', audio_source_path,  # Audio source input
                '-map', '0:v:0', # Video from input 0
                '-map', '1:a:0', # Audio from input 1
                '-c:v', 'copy',  # Copy video stream (faster if no filter applied)
                '-c:a', 'aac',   # Re-encode audio
                '-b:a', '128k',
                '-shortest',    # Ensure output duration matches shortest input
                '-y',
                output_path
            ]
            print("Running FFmpeg command (no subtitles, adding audio):")
            cmd_string = ' '.join([str(arg) for arg in ffmpeg_command_no_subs])
            print(f"Command: {cmd_string}")
            process = subprocess.run(ffmpeg_command_no_subs, check=True, capture_output=True, text=True)
            print(f"Successfully muxed audio into: {output_path}")
            return True # Return true as the operation (adding audio) succeeded

        # Write the ASS content to the current directory
        with open(temp_ass_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)

        print(f"Generated subtitle file: {temp_ass_path}")

        # FFmpeg command using two inputs and mapping streams
        ffmpeg_command = [
            'ffmpeg',
            '-i', vertical_video_path,  # Input 0: Vertically cropped video (silent)
            '-i', audio_source_path,   # Input 1: Original segment (with audio)
            # Use absolute path for subtitles file to avoid potential issues with ffmpeg's working directory
            '-filter_complex', f"[0:v]ass='{os.path.abspath(temp_ass_path)}'[video_out]",
            '-map', '[video_out]',     # Map the filtered video stream
            '-map', '1:a:0',           # Map the audio stream from input 1
            '-c:v', 'libx264',
            '-crf', '23',
            '-preset', 'medium',
            '-c:a', 'aac',           # Re-encode audio (required when filtering/mapping)
            '-b:a', '128k',
            '-shortest',             # Finish encoding when the shortest input ends
            '-y',
            output_path
        ]

        # Print the command for debugging
        print("Running FFmpeg command (burning subtitles and adding audio):")
        cmd_string = ' '.join([str(arg) for arg in ffmpeg_command])
        print(f"Command: {cmd_string}")

        # Run FFmpeg with the new command
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Successfully burned captions and added audio into: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"An error occurred during caption burning: {e}")
        return False
    finally:
        # Always clean up the subtitle file
        if os.path.exists(temp_ass_path):
            try:
                os.remove(temp_ass_path)
                print(f"Removed temporary subtitle file: {temp_ass_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary subtitle file: {e}")


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

def animate_captions(vertical_video_path, audio_source_path, transcription_result, output_path, style_cfg=None, highlight_meta=None):
    """Creates a video with word-by-word highlighted captions based on segments.
    - tone/keywords heuristic via optional highlight_meta for accent coloring
    """
    temp_animated_video = output_path + "_temp_anim.mp4"
    success = False
    cap = None
    out = None

    # Локальный helper: HEX -> RGBA (tuple)
    def _hex_to_rgba(hex_str: str, default=(255, 255, 0, 255)):
        try:
            s = str(hex_str).strip()
            if not s.startswith("#"):
                raise ValueError("No #")
            s = s[1:]
            if len(s) == 6:
                rr, gg, bb = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
                aa = 255
            elif len(s) == 8:
                rr, gg, bb, aa = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16)
            else:
                raise ValueError("Bad length")
            return (rr, gg, bb, aa)
        except Exception:
            return default

    # Посимвольная отрисовка с letter-spacing (упрощение для одной строки)
    def _draw_text_with_spacing(draw_obj, start_xy, text, font, fill, stroke_width=0, stroke_fill=None, spacing_px=0):
        x, y = start_xy
        for idx, ch in enumerate(text):
            draw_obj.text((x, y), ch, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
            # Используем getlength для более точного измерения ширины символа
            try:
                ch_w = font.getlength(ch)
            except AttributeError:  # Fallback for older/different PIL versions
                bbox = font.getbbox(ch) if hasattr(font, 'getbbox') else draw_obj.textbbox((0, 0), ch, font=font)
                ch_w = (bbox[2] - bbox[0]) if bbox else font.getsize(ch)[0]
            x += ch_w + (spacing_px if idx < len(text) - 1 else 0)

    # Оценка ширины текста с letter-spacing
    def _measure_text_width(draw_obj, text, font, spacing_px=0):
        if not text:
            return 0
        try:
            # Предпочтительный, более точный метод
            base_width = font.getlength(text)
            total_spacing = (len(text) - 1) * spacing_px if len(text) > 1 else 0
            return base_width + total_spacing
        except AttributeError:
            # Fallback для старых версий PIL или шрифтов без getlength
            total = 0
            for idx, ch in enumerate(text):
                bbox = font.getbbox(ch) if hasattr(font, 'getbbox') else draw_obj.textbbox((0, 0), ch, font=font)
                ch_w = (bbox[2] - bbox[0]) if bbox else font.getsize(ch)[0]
                total += ch_w
            total += (len(text) - 1) * spacing_px if len(text) > 1 else 0
            return total

    try:
        # --- Font Setup (Pillow) ---
        font_size = 38  # default legacy
        if style_cfg is not None:
            try:
                font_size = int(getattr(style_cfg, "font_size_px", font_size) or font_size)
            except Exception:
                pass

        font_path = fonts_path("Montserrat-Bold.ttf")
        font = None
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                print(f"Successfully loaded font: {font_path}")
            else:
                print(f"Font file not found at {font_path}. Using PIL default font.")
                font = ImageFont.load_default()
        except Exception as e:
            print(f"Warning: could not load TTF font at {font_path}: {e}. Using PIL default font.")
            font = ImageFont.load_default()

        # --- Pre-filter segments ---
        original_segments = transcription_result.get("segments", [])
        filtered_segments = [seg for seg in original_segments if seg.get('text', '').strip() != '[*]']
        if not filtered_segments:
            print("Warning: No non-[*] segments found in transcription. Captions might be empty.")

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
        # Base color
        text_rgba = _hex_to_rgba(getattr(style_cfg, "base_color", "#FFFF00FF") if style_cfg else "#FFFF00FF", (255, 255, 0, 255))
        stroke_color_rgb = (0, 0, 0)    # Black outline (legacy)
        stroke_width = 1                # Outline width in pixels

        # accent color mapping (palette from style_cfg or defaults)
        tone_color_rgba = None
        kw_set = set()
        if style_cfg is not None and highlight_meta:
            # «accent color mapping»
            try:
                # Prepare palette
                default_palette = {"urgency": "#FFD400", "drama": "#FF3B30", "positive": "#34C759"}
                pal_obj = getattr(style_cfg, "accent_palette", None)
                if pal_obj:
                    palette = {
                        "urgency": getattr(pal_obj, "urgency", default_palette["urgency"]),
                        "drama": getattr(pal_obj, "drama", default_palette["drama"]),
                        "positive": getattr(pal_obj, "positive", default_palette["positive"]),
                    }
                else:
                    palette = default_palette
                tone = str(highlight_meta.get("tone", "neutral") or "neutral")
                if tone in ("urgency", "drama", "positive"):
                    tone_hex = palette.get(tone)
                    if isinstance(tone_hex, str):
                        tone_color_rgba = _hex_to_rgba(tone_hex, text_rgba)
                # Build keyword set (lowercase)
                kws = highlight_meta.get("keywords", []) or []
                if isinstance(kws, (list, tuple)):
                    kw_set = {str(k).lower() for k in kws if isinstance(k, str)}
            except Exception:
                tone_color_rgba = None
                kw_set = set()

        # Shadow config
        sx = sy = 0
        shadow_rgba = (0, 0, 0, 128)
        if style_cfg is not None:
            sh = getattr(style_cfg, "shadow", None)
            if sh:
                try:
                    sx = int(getattr(sh, "x_px", 2) or 0)
                    sy = int(getattr(sh, "y_px", 2) or 0)
                except Exception:
                    sx = sy = 0
                shadow_rgba = _hex_to_rgba(getattr(sh, "color", "#00000080"), (0, 0, 0, 128))

        # Letter-spacing
        letter_spacing_px = 0
        if style_cfg is not None:
            try:
                letter_spacing_px = int(round(float(getattr(style_cfg, "letter_spacing_px", 0) or 0)))
            except Exception:
                letter_spacing_px = 0

        # Positioning
        position_mode = "safe_bottom"
        bottom_offset_pct = 22
        center_offset_pct = 12
        boundary_padding_px = 10
        if style_cfg is not None:
            pos = getattr(style_cfg, "position", None)
            if pos:
                position_mode = getattr(pos, "mode", "safe_bottom")
                try:
                    bottom_offset_pct = int(getattr(pos, "bottom_offset_pct", 22))
                except (ValueError, TypeError):
                    bottom_offset_pct = 22
                try:
                    center_offset_pct = int(getattr(pos, "center_offset_pct", 12))
                except (ValueError, TypeError):
                    center_offset_pct = 12
                try:
                    boundary_padding_px = int(getattr(pos, "boundary_padding_px", 10))
                except (ValueError, TypeError):
                    boundary_padding_px = 10

        # Emoji config and font (best-effort)
        emoji_enabled = False
        emoji_list_prepared = []
        emoji_max = 0
        emoji_window_s = 1.0  # emoji timing window ~1s from segment start
        emoji_font_size = max(8, int(font_size * 0.9))
        emoji_font = None

        if style_cfg is not None and isinstance(highlight_meta, dict):
            try:
                em_cfg = getattr(style_cfg, "emoji", None)
                emoji_enabled = bool(getattr(em_cfg, "enabled", False)) if em_cfg else False
                if emoji_enabled:
                    raw = list(highlight_meta.get("emojis", []) or [])
                    # дополнительная страховка по лимиту
                    emoji_max = int(getattr(em_cfg, "max_per_short", 0) or 0)
                    if emoji_max > 0:
                        emoji_list_prepared = [str(x) for x in raw if isinstance(x, str)][:emoji_max]
                    else:
                        emoji_list_prepared = []
            except Exception:
                emoji_enabled = False
                emoji_list_prepared = []

        def _load_emoji_font(sz: int):
            # emoji font loading (best-effort, platform paths)
            candidates = [
                "C:/Windows/Fonts/seguiemj.ttf",                        # Windows
                "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",    # Linux
                "/System/Library/Fonts/Apple Color Emoji.ttc",          # macOS
            ]
            for pth in candidates:
                try:
                    if os.path.exists(pth):
                        return ImageFont.truetype(pth, sz)
                except Exception:
                    continue
            return None

        if emoji_enabled:
            try:
                emoji_font = _load_emoji_font(emoji_font_size)
            except Exception:
                emoji_font = None

        def _apply_emojis(base_img, start_x_val, total_w_val, start_y_val):
            """
            Отрисовать эмодзи на отдельном RGBA-слое и скомпозитить поверх кадра.
            - emoji: heuristics and placement
            - emoji font loading (best-effort, platform paths)
            - emoji timing window ~1s from segment start
            - effects: "pulse" / "shiny" в первые 0.8с от старта сегмента, затем статика до 1.0с
            """
            try:
                if not (emoji_enabled and emoji_list_prepared):
                    return base_img

                # easing helpers for emoji
                def _clamp01(v):
                    try:
                        v = float(v)
                    except Exception:
                        return 0.0
                    if v < 0.0:
                        return 0.0
                    if v > 1.0:
                        return 1.0
                    return v

                def _ease_out_cubic(t):
                    # easing: easeOutCubic
                    return 1.0 - (1.0 - float(t)) ** 3

                def _ease_in_out_sine(t):
                    # easing: easeInOutSine
                    return 0.5 * (1.0 - math.cos(math.pi * float(t)))

                def _lerp(a, b, t):
                    return a + (b - a) * float(t)

                # Время сегмента для окна эффекта
                seg_t0 = 0.0
                if isinstance(active_segment, dict):
                    try:
                        seg_t0 = float(active_segment.get("start", 0.0) or 0.0)
                    except Exception:
                        seg_t0 = 0.0

                # Общее окно показа эмодзи ~1.0 c от начала сегмента (совместимость)
                dt = current_time - seg_t0
                if not (0.0 <= dt <= emoji_window_s):
                    # Вне окна показа — ведём себя как раньше (не показываем)
                    return base_img

                # Нормированное время эффекта
                effect_duration = 0.8
                u_raw = dt / (effect_duration if effect_duration > 0 else 1e-6)
                u = _clamp01(u_raw)

                # Чтение стиля эмодзи
                em_style = "none"
                if style_cfg is not None:
                    try:
                        em_cfg = getattr(style_cfg, "emoji", None)
                        em_style = str(getattr(em_cfg, "style", "none") or "none").lower() if em_cfg else "none"
                    except Exception:
                        em_style = "none"

                overlay_em = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                draw_em = ImageDraw.Draw(overlay_em)
                dx = int(font_size * 0.6)
                x_cursor = int(start_x_val + total_w_val + dx)
                y_draw = int(start_y_val)
                used_font = emoji_font if emoji_font else font  # fallback: текущий текстовый шрифт
                gap = max(2, int(emoji_font_size * 0.15))
                any_drawn = False

                for emo in emoji_list_prepared:
                    try:
                        # Базовые измерения
                        bbox = draw_em.textbbox((0, 0), emo, font=used_font)
                        w_emo = (bbox[2] - bbox[0]) if bbox else used_font.getsize(emo)[0]
                        h_emo = (bbox[3] - bbox[1]) if (bbox and len(bbox) >= 4) else font_size

                        # Рендер эмодзи на отдельный слой
                        emoji_layer = Image.new("RGBA", (max(1, w_emo), max(1, h_emo)), (0, 0, 0, 0))
                        emoji_draw = ImageDraw.Draw(emoji_layer)
                        emoji_draw.text((0, 0), emo, font=used_font, fill=(255, 255, 255, 255))

                        scale = 1.0
                        alpha_mult = 1.0

                        if em_style == "pulse":
                            # emoji: pulse effect (scale + alpha)
                            if 0.0 < u < 1.0:
                                u_e = _ease_in_out_sine(u)
                                scale = 1.0 + 0.06 * math.sin(math.pi * u_e)
                                alpha_mult = 0.85 + 0.15 * u
                            else:
                                # Вне окна эффекта, но всё ещё в окне показа (0.8..1.0) — статика
                                scale = 1.0
                                alpha_mult = 1.0

                        elif em_style == "shiny":
                            # emoji: shiny effect (moving gloss stripe)
                            if 0.0 < u < 1.0:
                                # Блик поверх исходного размера
                                gloss_width = max(3, int(0.15 * emoji_layer.width))
                                gloss_alpha = 0.45
                                progress = _ease_out_cubic(u)
                                gloss_x = int(round(_lerp(-gloss_width, emoji_layer.width, progress)))

                                gloss_layer = Image.new("RGBA", emoji_layer.size, (0, 0, 0, 0))
                                gloss_draw = ImageDraw.Draw(gloss_layer)
                                x0 = max(0, gloss_x)
                                x1 = min(emoji_layer.width, gloss_x + gloss_width)
                                if x1 > x0:
                                    gloss_color = (255, 255, 255, int(round(255 * gloss_alpha)))
                                    # Вертикальная полоса блика
                                    gloss_draw.rectangle([x0, 0, x1, emoji_layer.height], fill=gloss_color)

                                emoji_layer = Image.alpha_composite(emoji_layer, gloss_layer)

                                # Лёгкий масштаб
                                scale = 0.98 + 0.02 * u
                                alpha_mult = 1.0
                            else:
                                scale = 1.0
                                alpha_mult = 1.0

                        # Масштабирование слоя эмодзи
                        if abs(scale - 1.0) > 1e-3:
                            new_w = max(1, int(round(emoji_layer.width * scale)))
                            new_h = max(1, int(round(emoji_layer.height * scale)))
                            emoji_layer = emoji_layer.resize((new_w, new_h), resample=Image.BICUBIC)

                        # Применение альфа-множителя
                        if alpha_mult < 0.999:
                            r, g, b, a = emoji_layer.split()
                            a = a.point(lambda v, am=alpha_mult: int(v * am))
                            emoji_layer = Image.merge("RGBA", (r, g, b, a))

                        # Компоновка
                        overlay_em.paste(emoji_layer, (x_cursor, y_draw), mask=emoji_layer)
                        x_cursor += w_emo + gap
                        any_drawn = True

                    except Exception:
                        # Тихий пропуск проблемного эмодзи
                        continue

                if any_drawn:
                    return Image.alpha_composite(base_img, overlay_em)
                return base_img
            except Exception:
                return base_img

        # --- Calculate Fixed Y Position (after getting height) ---
        font_ascent = 0 # Default if font fails
        if font:
            try:
                font_ascent, _ = font.getmetrics()
            except Exception:
                 print("Warning: Could not get font metrics.")
        # legacy defaults
        bottom_margin_legacy = 120
        fixed_top_y_legacy = height - bottom_margin_legacy - font_ascent

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
            window_words = []  # [(text, word_info)] for per-word animation window
            if active_segment:
                segment_words_list = active_segment.get('words', [])
                num_words_in_segment = len(segment_words_list)

                if 0 <= active_word_idx_in_segment < num_words_in_segment:
                    word1_info = segment_words_list[active_word_idx_in_segment]
                    # Fallback to 'word' if 'text' missing (compat)
                    word1_text = (word1_info.get('text') or word1_info.get('word') or '').strip()

                    # Skip if the primary word is [*]
                    if word1_text != '[*]':
                        # Теперь отображаем только одно слово и в верхнем регистре
                        words_to_display = word1_text.upper().translate(str.maketrans('', '', string.punctuation))
                        window_words.append((words_to_display, word1_info))

            # --- Drawing Logic (Pillow - Max 2 words) ---
            # Draw only if we have a valid window and an active word within it
            if words_to_display and font: # Only draw if we have words and font loaded
                # Convert frame BGR OpenCV to RGB Pillow
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # При наличии style_cfg — поддерживаем RGBA + тень/альфа/letter-spacing,
                # иначе сохраняем прежнее поведение (совместимость).
                if style_cfg is None:
                    pil_image = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_image)
                    text_bbox = draw.textbbox((0, 0), words_to_display, font=font)
                    text_width = text_bbox[2] - text_bbox[0] if text_bbox else 0
                    start_x = (width - text_width) // 2
                    fixed_top_y = fixed_top_y_legacy
                    # Draw the text (legacy)
                    draw.text(
                        (start_x, fixed_top_y),
                        words_to_display,
                        font=font,
                        fill=(255, 255, 0),
                        stroke_width=stroke_width,
                        stroke_fill=stroke_color_rgb
                    )
                    drawn_any_text = True
                    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                else:
                    # Расширенный путь с наложением RGBA + опциональная анимация per-word
                    base_rgb = Image.fromarray(frame_rgb).convert("RGBA")
                    anim_cfg = getattr(style_cfg, "animate", None)

                    # Easing helpers + clamp (локальные)
                    def _clamp01(v):
                        try:
                            v = float(v)
                        except Exception:
                            return 0.0
                        if v < 0.0:
                            return 0.0
                        if v > 1.0:
                            return 1.0
                        return v

                    def _ease_out_cubic(t):
                        # easing: easeOutCubic
                        return 1.0 - (1.0 - float(t)) ** 3

                    # Общие измерения
                    tmp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
                    # Высота строки по bbox (вертикаль не зависит от spacing)
                    text_bbox = tmp_draw.textbbox((0, 0), words_to_display, font=font)
                    text_h = (text_bbox[3] - text_bbox[1]) if text_bbox else font_size

                    # Позиционирование
                    if position_mode == "center":
                        # Центрируем по вертикали, затем смещаем ниже центра
                        try:
                            offset_px = int(height * (center_offset_pct / 100.0))
                        except Exception:
                            offset_px = int(height * 0.12)  # 12% fallback
                        start_y = (height - text_h) // 2 + offset_px
                    else:  # safe_bottom
                        try:
                            margin_y = _compute_bottom_margin_px(height, bottom_offset_pct or 22)
                        except Exception:
                            margin_y = 120
                        start_y = height - margin_y - text_h # Используем text_h вместо font_ascent для большей точности

                    # --- Горизонтальное позиционирование и clamping ---
                    
                    # Общая ширина текста для центрирования
                    total_text_w = 0
                    
                    if not anim_cfg and not window_words:
                         total_text_w = _measure_text_width(tmp_draw, words_to_display, font, spacing_px=letter_spacing_px)
                    elif window_words:
                        space_w_bbox = tmp_draw.textbbox((0, 0), " ", font=font)
                        space_w = (space_w_bbox[2] - space_w_bbox[0]) if space_w_bbox else max(1, font_size // 3)
                        word_widths_local = [
                            _measure_text_width(tmp_draw, wt, font, spacing_px=letter_spacing_px)
                            for wt, _ in window_words
                        ]
                        total_text_w = sum(word_widths_local) + (len(word_widths_local) - 1) * space_w if word_widths_local else 0
                    
                    start_x = (width - total_text_w) // 2
                    
                    # Применяем boundary_padding_px для clamping
                    padding = boundary_padding_px or 10
                    start_x = max(padding, min(start_x, width - total_text_w - padding))
                    start_y = max(padding, min(start_y, height - text_h - padding))

                    if not anim_cfg:
                        import re as _re_kc
                        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                        draw_ov = ImageDraw.Draw(overlay)
                        
                        # Если window_words пуст, отрисовываем всю строку
                        if not window_words:
                            # Shadow
                            if sx != 0 or sy != 0 or (shadow_rgba and shadow_rgba[3] > 0):
                                _draw_text_with_spacing(draw_ov, (start_x + sx, start_y + sy), words_to_display, font, shadow_rgba, 0, None, letter_spacing_px)
                            # Main text
                            _draw_text_with_spacing(draw_ov, (start_x, start_y), words_to_display, font, text_rgba, stroke_width, stroke_color_rgb, letter_spacing_px)
                        else:
                            # Отрисовка по словам (для keyword coloring)
                            space_w_bbox = tmp_draw.textbbox((0, 0), " ", font=font)
                            space_w = (space_w_bbox[2] - space_w_bbox[0]) if space_w_bbox else max(1, font_size // 3)
                            word_widths = [_measure_text_width(tmp_draw, wt, font, spacing_px=letter_spacing_px) for wt, _ in window_words]
                            
                            # Shadow pass
                            if sx != 0 or sy != 0 or (shadow_rgba and shadow_rgba[3] > 0):
                                x_cursor = start_x
                                for idx, (w_text, _) in enumerate(window_words):
                                    _draw_text_with_spacing(draw_ov, (x_cursor + sx, start_y + sy), w_text, font, shadow_rgba, 0, None, letter_spacing_px)
                                    x_cursor += word_widths[idx] + (space_w if idx < len(word_widths) - 1 else 0)
                            
                            # Main pass
                            x_cursor = start_x
                            def _sanitize_kw(token: str) -> str: return _re_kc.sub(r"[^\wа-яё]+", "", token.lower())
                            for idx, (w_text, _) in enumerate(window_words):
                                use_rgba = text_rgba
                                if tone_color_rgba and kw_set and _sanitize_kw(w_text) in kw_set:
                                    use_rgba = tone_color_rgba
                                _draw_text_with_spacing(draw_ov, (x_cursor, start_y), w_text, font, use_rgba, stroke_width, stroke_color_rgb, letter_spacing_px)
                                x_cursor += word_widths[idx] + (space_w if idx < len(word_widths) - 1 else 0)

                        composed = Image.alpha_composite(base_rgb, overlay)
                        composed = _apply_emojis(composed, start_x, total_text_w, start_y)
                        drawn_any_text = True
                        frame = cv2.cvtColor(np.array(composed.convert("RGB")), cv2.COLOR_RGB2BGR)
                    else:
                        # --- Анимация per-word ---
                        try:
                            # Параметры анимации с клиппингом/фолбэками
                            anim_type = str(getattr(anim_cfg, "type", "slide-up") or "slide-up")
                            if anim_type not in ("slide-up", "pop-in"):
                                anim_type = "slide-up"
                            try:
                                duration_s = float(getattr(anim_cfg, "duration_s", 0.35) or 0.35)
                            except Exception:
                                duration_s = 0.35
                            duration_s = max(0.2, min(0.5, duration_s))
                            easing_name = str(getattr(anim_cfg, "easing", "easeOutCubic") or "easeOutCubic")
                            try:
                                stagger_ms = int(getattr(anim_cfg, "per_word_stagger_ms", 0) or 0)
                            except Exception:
                                stagger_ms = 0
                            stagger_s = max(0, stagger_ms) / 1000.0

                            # Выбор функции easing
                            if easing_name == "easeOutCubic":
                                ease = _ease_out_cubic
                            else:
                                ease = lambda t: float(t)  # Линейный фолбэк

                            # Подготовка окна слов; если не собрали — рендер одной строкой
                            if not window_words:
                                if words_to_display:
                                    window_words = [(words_to_display, {"start": current_time})]

                            # Измеряем ширины слов и пробела
                            space_w_bbox = tmp_draw.textbbox((0, 0), " ", font=font)
                            space_w = (space_w_bbox[2] - space_w_bbox[0]) if space_w_bbox else max(1, font_size // 3)
                            word_widths = [
                                _measure_text_width(tmp_draw, wt, font, spacing_px=letter_spacing_px)
                                for wt, _ in window_words
                            ]
                            total_text_w = sum(word_widths) + (len(word_widths) - 1) * space_w if word_widths else 0
                            start_x = (width - total_text_w) // 2

                            # Базовый вертикальный смещающий оффсет для slide-up
                            offsetY0 = int(max(12, min(24, round(0.25 * font_size))))

                            x_cursor = start_x
                            any_drawn_local = False

                            for idx, (w_text, w_info) in enumerate(window_words):
                                # t0 — время начала слова (сек)
                                try:
                                    t0 = float(w_info.get("start", current_time) or current_time)
                                except Exception:
                                    t0 = current_time
                                # Stagger по индексу в текущем окне
                                t0_staggered = t0 + idx * stagger_s

                                # Прогресс анимации per-word
                                progress_raw = (current_time - t0_staggered) / (duration_s if duration_s > 0 else 1e-6)
                                progress = _clamp01(progress_raw)
                                final = ease(progress)  # easing

                                # Преобразования: pop-in / slide-up
                                if anim_type == "pop-in":
                                    # pop-in: scale 0.85->1.0, alpha = final, translateY=0
                                    scale = 0.85 + 0.15 * final
                                    translateY = 0
                                    alpha_mult = final
                                else:
                                    # slide-up: смещение снизу, лёгкий scale 0.98->1.0, alpha = final
                                    translateY = int(round((1.0 - final) * offsetY0))
                                    scale = 0.98 + 0.02 * final
                                    alpha_mult = final

                                # Если слово ещё не началось с учётом stagger — не показываем
                                if progress <= 0.0 or alpha_mult <= 0.0:
                                    x_cursor += word_widths[idx] + (space_w if idx < len(word_widths) - 1 else 0)
                                    continue

                                # Рендер слова на отдельном RGBA-слое
                                word_w = word_widths[idx]
                                # Паддинги под тень, чтобы не обрезать смещённую копию
                                pad_left = max(0, -sx)
                                pad_top = max(0, -sy)
                                pad_right = max(0, sx)
                                pad_bottom = max(0, sy)
                                wl_w = max(1, word_w + pad_left + pad_right)
                                wl_h = max(1, text_h + pad_top + pad_bottom)
                                word_layer = Image.new("RGBA", (wl_w, wl_h), (0, 0, 0, 0))
                                draw_wl = ImageDraw.Draw(word_layer)

                                # Тень (идёт теми же трансформациями)
                                if sx != 0 or sy != 0 or (shadow_rgba and shadow_rgba[3] > 0):
                                    _draw_text_with_spacing(
                                        draw_wl,
                                        (pad_left + sx, pad_top + sy),
                                        w_text,
                                        font=font,
                                        fill=shadow_rgba,
                                        stroke_width=0,
                                        stroke_fill=None,
                                        spacing_px=letter_spacing_px
                                    )

                                # Основной текст с «keyword-based coloring (fallback to base_color)»
                                def _sanitize_kw(token: str) -> str:
                                    import re as _re_kw
                                    return _re_kw.sub(r"[^\wа-яё]+", "", token.lower())

                                use_rgba = text_rgba
                                if tone_color_rgba and kw_set:
                                    token_norm = _sanitize_kw(w_text)
                                    if token_norm in kw_set:
                                        use_rgba = tone_color_rgba

                                _draw_text_with_spacing(
                                    draw_wl,
                                    (pad_left, pad_top),
                                    w_text,
                                    font=font,
                                    fill=use_rgba,
                                    stroke_width=stroke_width,
                                    stroke_fill=stroke_color_rgb,
                                    spacing_px=letter_spacing_px
                                )

                                # Масштабирование слоя
                                if abs(scale - 1.0) > 1e-3:
                                    new_w = max(1, int(round(word_layer.width * scale)))
                                    new_h = max(1, int(round(word_layer.height * scale)))
                                    word_layer = word_layer.resize((new_w, new_h), resample=Image.BICUBIC)

                                # Умножение альфа-канала
                                if alpha_mult < 0.999:
                                    r, g, b, a = word_layer.split()
                                    a = a.point(lambda v, am=alpha_mult: int(v * am))
                                    word_layer = Image.merge("RGBA", (r, g, b, a))

                                # Позиционирование и композитинг (paste с альфа-маской)
                                paste_x = int(round(x_cursor - pad_left * (scale if scale else 1.0)))
                                paste_y = int(round(start_y + translateY - pad_top * (scale if scale else 1.0)))
                                base_rgb.paste(word_layer, (paste_x, paste_y), mask=word_layer)
                                any_drawn_local = True

                                # Сдвиг курсора по X (учитываем пробел между словами)
                                x_cursor += word_w + (space_w if idx < len(word_widths) - 1 else 0)

                            if any_drawn_local:
                                drawn_any_text = True
                                # Apply emojis near text (if enabled)
                                base_rgb = _apply_emojis(base_rgb, start_x, total_text_w, start_y)

                            # Конверсия обратно в OpenCV BGR
                            frame = cv2.cvtColor(np.array(base_rgb.convert("RGB")), cv2.COLOR_RGB2BGR)

                        except Exception as _anim_ex:
                            # Фолбэк: отрисовка без анимации при ошибке
                            overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                            draw_ov = ImageDraw.Draw(overlay)
                            total_text_w = _measure_text_width(tmp_draw, words_to_display, font, spacing_px=letter_spacing_px)
                            start_x = (width - total_text_w) // 2

                            if sx != 0 or sy != 0 or (shadow_rgba and shadow_rgba[3] > 0):
                                _draw_text_with_spacing(
                                    draw_ov,
                                    (start_x + sx, start_y + sy),
                                    words_to_display,
                                    font=font,
                                    fill=shadow_rgba,
                                    stroke_width=0,
                                    stroke_fill=None,
                                    spacing_px=letter_spacing_px
                                )

                            _draw_text_with_spacing(
                                draw_ov,
                                (start_x, start_y),
                                words_to_display,
                                font=font,
                                fill=text_rgba,
                                stroke_width=stroke_width,
                                stroke_fill=stroke_color_rgb,
                                spacing_px=letter_spacing_px
                            )

                            composed = Image.alpha_composite(base_rgb, overlay)
                            # Apply emojis near text (if enabled)
                            composed = _apply_emojis(composed, start_x, total_text_w, start_y)
                            drawn_any_text = True
                            frame = cv2.cvtColor(np.array(composed.convert("RGB")), cv2.COLOR_RGB2BGR)

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