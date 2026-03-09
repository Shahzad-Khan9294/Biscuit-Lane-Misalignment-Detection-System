import cv2

def init_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    return cap, fps

def init_video_writer(output_path, fourcc, fps, frame_shape):
    import cv2
    return cv2.VideoWriter(output_path, fourcc, fps, frame_shape)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates: x={x}, y={y}")