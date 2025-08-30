import cv2
import numpy as np
from moviepy.editor import *
import subprocess
import os
# Note: detect_faces_and_speakers and Frames are no longer used by crop_to_vertical_static
# from Components.Speaker import detect_faces_and_speakers, Frames
global Fps

def crop_to_vertical_static(input_video_path, output_video_path):
    """Crops the video to a 9:16 aspect ratio using a static centered crop."""
    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None # Return None on failure

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_height == 0 or fps == 0:
        print("Error: Video properties (height/fps) are invalid.")
        cap.release()
        return None

    # Calculate target 9:16 width based on original height
    vertical_height = original_height
    vertical_width = int(vertical_height * 9 / 16)

    # Ensure the calculated width is even (required by some codecs)
    if vertical_width % 2 != 0:
        vertical_width -= 1

    print(f"Original Dims: {original_width}x{original_height} @ {fps:.2f}fps")
    print(f"Target Vertical Dims: {vertical_width}x{vertical_height}")

    if original_width < vertical_width or vertical_width <= 0:
        print("Error: Original video width is less than the calculated vertical width or width is invalid.")
        cap.release()
        return None

    # Calculate static horizontal crop start/end points (centered)
    x_start = (original_width - vertical_width) // 2
    x_end = x_start + vertical_width
    print(f"Static Crop Range (Horizontal): {x_start} to {x_end}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4 output
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vertical_width, vertical_height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        cap.release()
        return None

    # Set global Fps (if needed elsewhere, otherwise consider removing global)
    global Fps
    Fps = fps

    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Apply the static crop
        cropped_frame = frame[:, x_start:x_end]

        # Basic check in case cropping resulted in unexpected shape
        if cropped_frame.shape[1] != vertical_width or cropped_frame.shape[0] != vertical_height:
             print(f"Warning: Cropped frame shape {cropped_frame.shape} doesn't match target {vertical_width}x{vertical_height}. Adjusting...")
             # Attempt to resize, though this indicates an issue upstream or with calculations
             cropped_frame = cv2.resize(cropped_frame, (vertical_width, vertical_height))

        out.write(cropped_frame)
        processed_frames += 1

    print(f"Processed {processed_frames}/{total_frames} frames.")
    cap.release()
    out.release()
    print(f"Static vertical cropping complete. Video saved to: {output_video_path}")
    return output_video_path # Return path on success


def crop_to_vertical(input_video_path, output_video_path):
    # detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vertical_height = int(original_height)
    vertical_width = int(vertical_height * 9 / 16)
    print(vertical_height, vertical_width)


    if original_width < vertical_width:
        print("Error: Original video width is less than the desired vertical width.")
        return

    x_start = (original_width - vertical_width) // 2
    x_end = x_start + vertical_width
    print(f"start and end - {x_start} , {x_end}")
    print(x_end-x_start)
    half_width = vertical_width // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vertical_width, vertical_height))
    global Fps
    Fps = fps
    print(fps)
    count = 0
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) >-1:
            if len(faces) == 0:
                (x, y, w, h) = Frames[count]

            # (x, y, w, h) = faces[0]
            try:
                #check if face 1 is active
                (X, Y, W, H) = Frames[count]
            except Exception as e:
                print(e)
                (X, Y, W, H) = Frames[count][0]
                print(Frames[count][0])

            for f in faces:
                x1, y1, w1, h1 = f
                center = x1+ w1//2
                if center > X and center < X+W:
                    x = x1
                    y = y1
                    w = w1
                    h = h1
                    break

            # print(faces[0])
            centerX = x+(w//2)
            print(centerX)
            print(x_start - (centerX - half_width))
            if count == 0 or (x_start - (centerX - half_width)) <1 :
                ## IF dif from prev fram is low then no movement is done
                pass #use prev vals
            else:
                x_start = centerX - half_width
                x_end = centerX + half_width


                if int(cropped_frame.shape[1]) != x_end- x_start:
                    if x_end < original_width:
                        x_end += int(cropped_frame.shape[1]) - (x_end-x_start)
                        if x_end > original_width:
                            x_start -= int(cropped_frame.shape[1]) - (x_end-x_start)
                    else:
                        x_start -= int(cropped_frame.shape[1]) - (x_end-x_start)
                        if x_start < 0:
                            x_end += int(cropped_frame.shape[1]) - (x_end-x_start)
                    print("Frame size inconsistant")
                    print(x_end- x_start)

        count += 1
        cropped_frame = frame[:, x_start:x_end]
        if cropped_frame.shape[1] == 0:
            x_start = (original_width - vertical_width) // 2
            x_end = x_start + vertical_width
            cropped_frame = frame[:, x_start:x_end]

        print(cropped_frame.shape)

        out.write(cropped_frame)

    cap.release()
    out.release()
    print("Cropping complete. The video has been saved to", output_video_path, count)


# --- New Function: Average Face Centered Crop ---

def crop_to_vertical_average_face(input_video_path, output_video_path, sample_interval_seconds=0.5):
    """Crops video to 9:16 based on the average horizontal face position sampled periodically."""
    print("Starting average face centered vertical crop...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return None

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_height <= 0 or fps <= 0:
        print("Error: Invalid video properties (height or fps <= 0).")
        cap.release()
        return None

    print(f"Input: {original_width}x{original_height} @ {fps:.2f}fps")

    # Calculate target 9:16 width
    vertical_height = 1280
    vertical_width = 720

    if original_width < vertical_width or vertical_width <= 0:
        print("Error: Original width too small for vertical crop.")
        cap.release()
        return None

    print(f"Target Vertical Dims: {vertical_width}x{vertical_height}")

    # --- First Pass: Sample face positions ---
    face_centers_x = []
    frames_to_skip = int(fps * sample_interval_seconds)
    if frames_to_skip < 1: frames_to_skip = 1 # Sample at least every frame if interval is too small

    frame_count = 0
    print(f"Sampling face position every {frames_to_skip} frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_to_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Adjust detection parameters if needed (e.g., scaleFactor, minNeighbors)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            if len(faces) > 0:
                # Assume the largest face is the main one if multiple are detected
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                centerX = x + w / 2
                face_centers_x.append(centerX)
                # Optional: Draw box on sample frame for debugging
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # cv2.imshow('Sample', frame); cv2.waitKey(1)

        frame_count += 1

    # --- Calculate Average Position ---
    average_face_center_x = None
    if face_centers_x:
        average_face_center_x = np.mean(face_centers_x)
        print(f"Found {len(face_centers_x)} face samples. Average center X: {average_face_center_x:.2f}")
    else:
        print("Warning: No faces detected during sampling. Falling back to frame center.")
        average_face_center_x = original_width / 2

    # --- Calculate Static Crop Box ---
    half_vertical_width = vertical_width // 2
    x_start = int(average_face_center_x - half_vertical_width)
    x_end = x_start + vertical_width

    # Clamp crop box to frame boundaries
    x_start = max(0, x_start)
    x_end = min(original_width, x_end)

    # Adjust x_start if clamping x_end changed the width
    if x_end - x_start != vertical_width:
         x_start = x_end - vertical_width
         x_start = max(0, x_start) # Re-clamp x_start just in case

    # Final check if width calculation is still correct after clamping
    if x_end - x_start != vertical_width:
        print(f"Error: Could not calculate valid crop window ({x_start}-{x_end}) for width {vertical_width}. Check logic.")
        cap.release()
        return None

    print(f"Calculated Static Crop Box: X = {x_start} to {x_end}")

    # --- Second Pass: Apply crop and write video ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind video capture

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vertical_width, vertical_height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        cap.release()
        return None

    print("Applying static crop and writing output video...")
    written_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[:, x_start:x_end]

        # Sanity check shape before writing (optional but good)
        if cropped_frame.shape[1] != vertical_width or cropped_frame.shape[0] != vertical_height:
            print(f"Warning: Frame {written_frames} cropped shape {cropped_frame.shape} != target {vertical_width}x{vertical_height}. Resizing.")
            cropped_frame = cv2.resize(cropped_frame, (vertical_width, vertical_height))

        out.write(cropped_frame)
        written_frames += 1

    print(f"Finished writing {written_frames} frames.")
    cap.release()
    out.release()
    print(f"Average face centered vertical crop complete. Saved to: {output_video_path}")
    return output_video_path


def crop_to_70_percent_with_blur(input_video_path, output_video_path):
    """
    Crops video to 70% of original width with 213:274 aspect ratio for content,
    then creates a 9:16 final frame with blurred background and centered content.
    Uses dynamic sizing based on original video height.
    """
    print("Starting 70% width crop with blur background (9:16 final aspect ratio)...")

    # Get video properties using ffprobe
    try:
        import json
        ffprobe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            input_video_path
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        original_width = probe_data['streams'][0]['width']
        original_height = probe_data['streams'][0]['height']
    except Exception as e:
        print(f"Error getting video dimensions: {e}")
        return None

    print(f"Original dimensions: {original_width}x{original_height}")

    # Calculate final 9:16 dimensions
    final_height = 1280
    final_width = 720

    # Ensure final dimensions are even (required by some codecs)
    if final_width % 2 != 0:
        final_width -= 1

    print(f"Final 9:16 dimensions: {final_width}x{final_height}")

    # Calculate 70% width crop for content (maintain 9:16 aspect ratio)
    content_width = int(original_width * 0.7)
    content_aspect_ratio = 9 / 16  # ≈ 0.5625
    content_height = min(int(content_width / content_aspect_ratio), original_height)

    # Ensure content dimensions are even
    if content_width % 2 != 0:
        content_width -= 1
    if content_height % 2 != 0:
        content_height -= 1

    print(f"Content crop dimensions (70% width, 9:16 aspect): {content_width}x{content_height}")

    # Calculate scaling for content to fit in final frame while maintaining aspect ratio
    scale_factor = min(final_width / content_width, final_height / content_height)
    scaled_content_width = int(content_width * scale_factor)
    scaled_content_height = int(content_height * scale_factor)

    # Ensure scaled dimensions are even
    if scaled_content_width % 2 != 0:
        scaled_content_width -= 1
    if scaled_content_height % 2 != 0:
        scaled_content_height -= 1

    print(f"Scaled content dimensions: {scaled_content_width}x{scaled_content_height}")

    # Calculate positioning for centering content in final frame
    content_x = (final_width - scaled_content_width) // 2
    content_y = (final_height - scaled_content_height) // 2

    print(f"Content positioning in final frame: x={content_x}, y={content_y}")

    # Calculate blur radius based on original dimensions
    blur_radius = min(original_width, original_height) / 20
    print(f"Blur radius: {blur_radius}")

    # Build filter_complex to eliminate black bars and use blurred background that fills the entire frame.
    # Background: scale with force_original_aspect_ratio=increase then crop to FINAL WxH, apply boxblur, setsar=1.
    # Foreground: center crop to 70% width (CROP_W x CROP_H), then scale to FINAL_CONTENT_W x FINAL_CONTENT_H, setsar=1.
    # Compose: overlay at computed offsets, finalize as yuv420p.
    filter_complex = (
        "[0:v]split=2[fg_src][bg_src];"
        f"[bg_src]scale={final_width}:{final_height}:force_original_aspect_ratio=increase,"
        f"crop={final_width}:{final_height},"
        f"boxblur=luma_radius={blur_radius}:luma_power=5:chroma_radius={blur_radius}:chroma_power=1,setsar=1[bg];"
        f"[fg_src]crop={content_width}:{content_height}:(iw-{content_width})/2:(ih-{content_height})/2,"
        f"scale={scaled_content_width}:{scaled_content_height}:force_original_aspect_ratio=decrease,setsar=1[fg];"
        f"[bg][fg]overlay={content_x}:{content_y}:shortest=1,format=yuv420p[vout]"
    )

    # FFmpeg command using new filter and mappings; keep current crf/preset values
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_video_path,
        '-filter_complex', filter_complex,
        '-map', '[vout]',
        '-map', '0:a?',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-preset', 'medium',
        '-c:a', 'copy',
        '-y',
        output_video_path
    ]

    # Logging: print and, if logger available, logger.debug
    try:
        from Components.Logger import logger  # type: ignore
    except Exception:
        logger = None

    print("FFmpeg filter_complex:", filter_complex)
    print("FFmpeg command:", ' '.join([str(a) for a in ffmpeg_cmd]))
    if logger:
        try:
            logger.debug(f"FFmpeg filter_complex: {filter_complex}")
            logger.debug(f"FFmpeg command: {' '.join([str(a) for a in ffmpeg_cmd])}")
        except Exception:
            pass

    try:
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"Successfully created 70% crop with blurred background video: {output_video_path}")
        if logger:
            try:
                logger.debug(f"ffmpeg stdout: {result.stdout}")
                logger.debug(f"ffmpeg stderr: {result.stderr}")
            except Exception:
                pass
        return output_video_path
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        if logger:
            try:
                logger.error(f"Error running FFmpeg: {e}")
                logger.error(f"FFmpeg stdout: {e.stdout}")
                logger.error(f"FFmpeg stderr: {e.stderr}")
            except Exception:
                pass
        return None
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        if logger:
            try:
                logger.error(f"An error occurred during processing: {e}")
            except Exception:
                pass
        return None
