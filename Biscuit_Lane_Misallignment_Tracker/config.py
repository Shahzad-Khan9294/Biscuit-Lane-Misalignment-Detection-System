import numpy as np

# Video paths and output
VIDEO_PATH = r'C:\Users\shahzad.khan\Documents\Biscuits_Lane_Tracker\Biscuit_Conveyor_Test_Videos\NVR_ch1_main_Trim_3.mp4'
OUTPUT_PATH = "lane_detection_output.mp4"

# Frame & video settings
TARGET_HEIGHT = 600
SKIP_TO_SECOND = 70
FRAMES_FOR_3_SEC = None  # Will be set after fps calculation
ROLLING_WINDOW_SIZE = 100
MAX_HISTORY_LENGTH = 100
EXPECTED_LANES = 24

# Lane colors
np.random.seed(42)
colors_24 = []
for i in range(EXPECTED_LANES):
    hue = int(i * 180 / EXPECTED_LANES)  # Distribute hues evenly
    color_hsv = np.uint8([[[hue, 255, 255]]])
    import cv2
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    colors_24.append(tuple(map(int, color_bgr)))

# Polygon ROI points
OFFSET = 4.5
POLYGON_POINTS = np.array([[266, 314], [652 - OFFSET, 312], [707 - OFFSET, 518], [233, 520]], dtype=np.int32)

# Lane analysis thresholds
CROSS_THRESHOLD = 1.5
MAX_GAP_THRESHOLD = 60
MAX_DEVIATION_THRESHOLD = 9
EXTREME_THRESHOLD = 0.2

# Regression smoothing
ALPHA = 0.05