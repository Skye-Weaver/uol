import cv2



def analyze_face_position_lightweight(video_path: str, sample_rate: int = 25) -> float:
    """
    Analyzes the video to find the average horizontal position of the largest face,
    sampling the video at a given frame rate to ensure performance.

    Args:
        video_path (str): The path to the video file.
        sample_rate (int): The interval at which to sample frames (e.g., 25 means every 25th frame).

    Returns:
        float: The average horizontal (X-coordinate) center of the largest face found.
               Returns the center of the frame if no faces are detected.
    """
    import cv2
    
    # Use the provided Haar Cascade file for face detection
    from .config import get_config
    cfg = get_config()
    face_cascade = cv2.CascadeClassifier(cfg.paths.face_cascade)
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier. Make sure 'haarcascade_frontalface_default.xml' is in the correct path.")
        return 0  # Return a default value

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return 0

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    face_x_coords = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # Find the largest face by area (w*h)
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                
                # Calculate the center of the face on the X-axis
                center_x = x + w / 2
                face_x_coords.append(center_x)

        frame_count += 1

    cap.release()

    if face_x_coords:
        # Return the average of all collected face center coordinates
        return sum(face_x_coords) / len(face_x_coords)
    else:
        # If no faces were found, return the horizontal center of the video
        print("Warning: No faces detected in the video. Returning the frame center.")
        return video_width / 2
