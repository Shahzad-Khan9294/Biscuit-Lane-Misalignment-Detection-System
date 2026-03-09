import os
import cv2
import winsound
import numpy as np
from collections import deque


np.random.seed(42)  # Generate 24 unique colors for 24 columns
colors_24 = []
for i in range(24):
    hue = int(i * 180 / 24)  # Distribute hues evenly
    color_hsv = np.uint8([[[hue, 255, 255]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    colors_24.append(tuple(map(int, color_bgr)))

# Mouse callback function to print coordinates
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates: x={x}, y={y}")

# Load video file
# video_path = 'biscuits_vid1.mp4'
video_path = 'Biscuit_Conveyor_Test_Videos\\NVR_ch1_main_Trim_2.mp4'
cap = cv2.VideoCapture(video_path)
# Video Writer Setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = "lane_detection_output.mp4"
out = None

if not cap.isOpened():
    exit()

# Get video frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
frames_for_3_sec = int(fps * 0)
skip_to_second = 70

# Define polygon ROI points
offset = 4.5
polygon_points = np.array([[266, 314], [652 - offset, 312], [707 - offset, 518], [233, 520]], dtype=np.int32)

# Set up mouse callback
cv2.namedWindow('Frame with Contours')
cv2.setMouseCallback('Frame with Contours', mouse_callback)

frame_count = 0
trajectory_image = None
column_boundaries = None
collecting_trajectories = True

# Misalignment tracking
misaligned_frames = set()
misalignment_triggered = False
rolling_window_size = 260   # Greater Value More Lane Consistency (200-300)
trajectory_queue = []

# Lane history for slope-preserving tracking
EXPECTED_LANES = 24
lane_history = [[] for _ in range(EXPECTED_LANES)]
max_history_length = 100

lane_misalignment_history = {}   # persistent across frames
# Create folder for misaligned images
misaligned_output_dir = "misalligned_output_images"
os.makedirs(misaligned_output_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 2 != 0:
        continue

    height, width = frame.shape[:2]
    target_height = 600
    aspect_ratio = width / height
    target_width = int(target_height * aspect_ratio)
    frame = cv2.resize(frame, (target_width, target_height))
    # Initialize video writer once (after knowing frame size)
    if out is None:
        out = cv2.VideoWriter(output_path, fourcc, fps / 2, (target_width, target_height))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    otsu_thresh = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)

    mask = np.zeros_like(otsu_thresh)
    cv2.fillPoly(mask, [polygon_points], 255)
    roi_thresh = cv2.bitwise_and(otsu_thresh, mask)

    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 3 and h >= 3 and w < 100 and h < 100:
            filtered_contours.append(contour)

    frame_with_contours = frame.copy()
    cv2.polylines(frame_with_contours, [polygon_points], isClosed=True, color=(0, 255, 255), thickness=1)

    shift_right = 7
    shift_left = 5   #6
    top_left, top_right, bottom_right, bottom_left = polygon_points
    cv2.line(frame_with_contours,
             (int(top_left[0] + shift_left), int(top_left[1])),
             (int(bottom_left[0] + shift_left), int(bottom_left[1])),
             (0, 0, 0), 1)
    cv2.line(frame_with_contours,
             (int(top_right[0] - shift_right), int(top_right[1])),
             (int(bottom_right[0] - shift_right), int(bottom_right[1])),
             (0, 0, 0), 1)

    left_x1, left_y1 = int(top_left[0] + shift_left), int(top_left[1])
    left_x2, left_y2 = int(bottom_left[0] + shift_left), int(bottom_left[1])
    right_x1, right_y1 = int(top_right[0] - shift_right), int(top_right[1])
    right_x2, right_y2 = int(bottom_right[0] - shift_right), int(bottom_right[1])

    if trajectory_image is None:
        trajectory_image = np.zeros((target_height, target_width), dtype=np.uint8)

    # Collect initial trajectories
    if collecting_trajectories and frame_count < frames_for_3_sec:
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x2 = x + w
            y_center = y + h // 2
            cv2.circle(trajectory_image, (x2, y_center), 1, 255, -1)

    elif collecting_trajectories and frame_count == frames_for_3_sec:
        collecting_trajectories = False
        cv2.imshow('Trajectory Image', trajectory_image)
        cv2.waitKey(1000)

        contours_traj, _ = cv2.findContours(trajectory_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fitted_lines = []
        if contours_traj:
            for contour in contours_traj:
                if len(contour) >= 10:
                    # vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    contour = cv2.convexHull(contour)
                    vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01) # DIST_L1, DIST_L2, DIST_HUBER, DIST_FAIR, DIST_WELSCH
                    M = cv2.moments(contour)
                    avg_x = M["m10"]/M["m00"] if M["m00"] != 0 else x0[0]
                    contour_length = cv2.arcLength(contour, False)
                    fitted_lines.append((avg_x, contour_length, (vx[0], vy[0], x0[0], y0[0])))
            fitted_lines.sort(key=lambda x: x[1], reverse=True)
            min_x_distance = 10
            selected_lines = []
            for line_info in fitted_lines:
                avg_x, length, line_params = line_info
                if all(abs(avg_x - s[0]) >= min_x_distance for s in selected_lines):
                    selected_lines.append(line_info)
                if len(selected_lines) >= EXPECTED_LANES:
                    break
            selected_lines.sort(key=lambda x: x[0])
            column_boundaries = [(avg_x, (vx, vy, x0, y0)) for avg_x, _, (vx, vy, x0, y0) in selected_lines]

        if skip_to_second > 0:
            skip_to_frame = int(skip_to_second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_to_frame)
            frame_count = skip_to_frame
            print(f"Skipped to {skip_to_second} seconds (frame {skip_to_frame})")

    # Rolling window update & slope-preserving lane history
    current_frame_points = [(x + w, y + h // 2) for x, y, w, h in [cv2.boundingRect(c) for c in filtered_contours]]
    trajectory_queue.append(current_frame_points)
    if len(trajectory_queue) > rolling_window_size:
        trajectory_queue.pop(0)

    if trajectory_queue:
        trajectory_image = np.zeros((target_height, target_width), dtype=np.uint8)
        for traj_frame in trajectory_queue:
            for x2, y_center in traj_frame:
                cv2.circle(trajectory_image, (x2, y_center), 1, 255, -1)
        contours_traj, _ = cv2.findContours(trajectory_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fitted_lines = []
        if contours_traj:
            for contour in contours_traj:
                if len(contour) >= 10:
                    # vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    contour = cv2.convexHull(contour)
                    vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01) #DIST_L1, DIST_L2, DIST_HUBER, DIST_FAIR, DIST_WELSCH
                    M = cv2.moments(contour)
                    avg_x = M["m10"]/M["m00"] if M["m00"] != 0 else x0[0]
                    contour_length = cv2.arcLength(contour, False)
                    fitted_lines.append((avg_x, contour_length, (vx[0], vy[0], x0[0], y0[0])))
            fitted_lines.sort(key=lambda x: x[1], reverse=True)
            min_x_distance = 10
            selected_lines = []
            for line_info in fitted_lines:
                avg_x, length, line_params = line_info
                if all(abs(avg_x - s[0]) >= min_x_distance for s in selected_lines):
                    selected_lines.append(line_info)
                if len(selected_lines) >= EXPECTED_LANES:
                    break
            selected_lines.sort(key=lambda x: x[0])
            # Update lane history
            for i in range(EXPECTED_LANES):
                if i < len(selected_lines):
                    _, _, line_params = selected_lines[i]
                    lane_history[i].append(line_params)
                    if len(lane_history[i]) > max_history_length:
                        lane_history[i].pop(0)
            # Median-based slope-preserving columns 
            column_boundaries = []
            alpha = 0.01            # smaller = more stable (0.05 even more stable)
            smoothed_lanes = [None] * EXPECTED_LANES
            for i in range(EXPECTED_LANES):
                if lane_history[i]:
                    vx, vy, x0, y0 = np.median(lane_history[i], axis=0)
                    if smoothed_lanes[i] is None:
                        smoothed_lanes[i] = np.array([vx, vy, x0, y0])
                    else:
                        smoothed_lanes[i] = (alpha * np.array([vx, vy, x0, y0]) + (1 - alpha) * smoothed_lanes[i])
                    vx_s, vy_s, x0_s, y0_s = smoothed_lanes[i]
                    column_boundaries.append((x0_s, (vx_s, vy_s, x0_s, y0_s)))
                else:
                    column_boundaries.append((0, (0,1,0,0)))

    # Compute centroids 
    centroids_x, centroids_y = [], []
    for contour in filtered_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w//2, y + h//2
        centroids_x.append(cx)
        centroids_y.append(cy)

    # Assign centroids & check misalignment
    column_labels = []
    misaligned_lanes = {}
    cross_threshold = 0.5
    if column_boundaries and filtered_contours:
        for i, (cx, cy) in enumerate(zip(centroids_x, centroids_y)):
            col = 0
            for j, (avg_x, (vx, vy, x0, y0)) in enumerate(column_boundaries):
                line_x_at_cy = x0 + (cy - y0) * (vx / vy) if abs(vy) > 1e-3 else x0
                if cx <= line_x_at_cy:
                    col = j
                    break
            else:
                col = EXPECTED_LANES - 1
            column_labels.append(col)
            avg_x, (vx, vy, x0, y0) = column_boundaries[col]
            line_x_at_cy = x0 + (cy - y0) * (vx / vy) if abs(vy) > 1e-3 else x0
            diff = cx - line_x_at_cy
            if diff > cross_threshold:
                if col not in misaligned_lanes:
                    misaligned_lanes[col] = []
                misaligned_lanes[col].append((cx, cy))
        # If any misalignment detected in this frame, save it
        if misaligned_lanes:
            image_name = f"misaligned_frame_{frame_count}.jpg"
            save_path = os.path.join(misaligned_output_dir, image_name)
            cv2.imwrite(save_path, frame_with_contours)
    
    # ---- PER LANE ADAPTIVE SLOPE LEVEL INTER LANE STRIP ALLIGNMENT CHECK CODE + TEMPORAL FILTER VERSION----
    max_horizontal_deviation = 5     # base allowed deviation
    margin_factor = 0.01             # slope influence tuning
    temporal_window = 3              # number of frames to track
    temporal_trigger_count = 3       # must misalign in N frames to beep
    lane_centroids = {}
    lane_boundaries = {}
    lane_thresholds = {}
    # Step 1: Define lane boundaries + compute slope-based thresholds
    for i, (avg_x, (vx, vy, x0, y0)) in enumerate(column_boundaries):
        # Compute lane boundaries
        if i == 0:
            left_x = 0
        else:
            left_x = int((column_boundaries[i-1][0] + avg_x) / 2)
        if i == len(column_boundaries)-1:
            right_x = frame_with_contours.shape[1]
        else:
            right_x = int((avg_x + column_boundaries[i+1][0]) / 2)
        lane_boundaries[i] = (left_x, right_x)
        slope = (vx / vy) if abs(vy) > 1e-3 else 0
        lane_thresholds[i] = max_horizontal_deviation + abs(slope) * margin_factor
        if i not in lane_misalignment_history:
            lane_misalignment_history[i] = deque(maxlen=temporal_window)
    # Step 2: Assign centroids to lanes
    for idx, (cx, cy) in enumerate(zip(centroids_x, centroids_y)):
        lane_idx = column_labels[idx]
        left_x, right_x = lane_boundaries[lane_idx]
        if left_x <= cx <= right_x:
            lane_centroids.setdefault(lane_idx, []).append((cx, cy))
    # Step 3: Check misalignment per lane
    for lane_idx, points in lane_centroids.items():
        bad_count = 0
        if len(points) < 2:
            lane_misalignment_history[lane_idx].append(False)
            continue
        points_sorted = sorted(points, key=lambda p: p[1])
        avg_x, (vx, vy, x0, y0) = column_boundaries[lane_idx]
        lane_thresh = lane_thresholds[lane_idx]
        misalignment_detected = False
        # Precompute lane normal vector (IMPORTANT FIX)
        if abs(vy) > 1e-3:
            lane_dx = vx
            lane_dy = vy
        else:
            lane_dx = 0
            lane_dy = 1
        norm = (lane_dx**2 + lane_dy**2) ** 0.5
        lane_dx /= norm
        lane_dy /= norm
        # Normal direction (perpendicular)
        normal_x = -lane_dy
        normal_y = lane_dx
        for i in range(len(points_sorted) - 1):
            (cx1, cy1), (cx2, cy2) = points_sorted[i], points_sorted[i+1]
            dx = cx2 - cx1
            dy = cy2 - cy1
            # Project displacement onto lane normal
            dx_along_lane = abs(dx * normal_x + dy * normal_y)
            if dx_along_lane > lane_thresh:
                bad_count += 1  # misalignment_detected = True   
                # Visual markers
                cv2.circle(frame_with_contours, (cx1, cy1), 5, (0, 0, 255), -1)
                cv2.circle(frame_with_contours, (cx2, cy2), 5, (0, 0, 255), -1)
                warning_text = f"Misaligned in lane {lane_idx}"
                cv2.putText(frame_with_contours, warning_text,
                            (10, 120 + 20 * lane_idx),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            # Visualization line
            cv2.line(frame_with_contours,
                    (cx1, cy1), (cx2, cy2),
                    colors_24[lane_idx], 1)
        if bad_count >= 2:
            misalignment_detected = True
        # Step 4: Rolling Temporal Filter
        lane_misalignment_history[lane_idx].append(misalignment_detected)
        if lane_misalignment_history[lane_idx].count(True) >= temporal_trigger_count:
            if frame_count not in misaligned_frames:
                misaligned_frames.add(frame_count)
                print(f"Misalignment detected at frame {frame_count} in lane {lane_idx}")
                image_name = f"misalignment_frame_{frame_count}_lane_{lane_idx}.png"
                cv2.imwrite(os.path.join(misaligned_output_dir, image_name),
                            frame_with_contours)
                winsound.Beep(1800, 400)
                    
        # Draw lanes preserving slope
        for idx, (avg_x, (vx, vy, x0, y0)) in enumerate(column_boundaries):
            if abs(vy) > 0.001:
                t1 = -y0 / vy
                x1 = int(x0 + t1*vx)
                y1 = 0
                t2 = (target_height - y0)/vy
                x2 = int(x0 + t2*vx)
                y2 = target_height
            else:
                x1, y1, x2, y2 = int(x0), 0, int(x0), target_height
            color = (0,0,255) if idx in misaligned_lanes else colors_24[idx]
            cv2.line(frame_with_contours, (x1, y1), (x2, y2), color, 1)

    # Draw contours & centroids
    for i, contour in enumerate(filtered_contours):
        lane_idx = column_labels[i]
        cv2.drawContours(frame_with_contours, [contour], -1, colors_24[lane_idx], 1)
        cv2.circle(frame_with_contours, (centroids_x[i], centroids_y[i]), 1, (0,0,255), -1)

    # Extreme left/right checks 
    extreme_crossed = {"left": [], "right": []}
    extreme_threshold = 0.2 # pixels beyond black line
    for i, (cx, cy) in enumerate(zip(centroids_x, centroids_y)):
        lane_idx = column_labels[i]
        if lane_idx == 0:
            left_line_x = left_x1 + (cx - left_x1) * 0 if abs(left_y2 - left_y1) < 1e-3 else left_x1 + (cy - left_y1) / (left_y2 - left_y1) * (left_x2 - left_x1)
            if cx < left_line_x - extreme_threshold:
                extreme_crossed["left"].append((cx, cy))
        if lane_idx == len(column_boundaries)-1:
            right_line_x = right_x1 + (cx - right_x1) * 0 if abs(right_y2 - right_y1) < 1e-3 else right_x1 + (cy - right_y1) / (right_y2 - right_y1) * (right_x2 - right_x1)
            if cx > right_line_x + extreme_threshold:
                extreme_crossed["right"].append((cx, cy))
    if extreme_crossed["left"] or extreme_crossed["right"]:
        warning_text = "Extreme Boundary Cross: "
        if extreme_crossed["left"]:
            warning_text += "LEFT "
        if extreme_crossed["right"]:
            warning_text += "RIGHT"
        cv2.putText(frame_with_contours, warning_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        print(f"Extreme boundary crossing at frame {frame_count} | {warning_text}")
        if frame_count not in misaligned_frames:
            misaligned_frames.add(frame_count)
            image_name = f"extreme_misaligned_frame_{frame_count}.png"
            save_path = os.path.join(misaligned_output_dir, image_name)
            cv2.imwrite(save_path, frame_with_contours)
            winsound.Beep(1500,300)

    # Timestamp & Frame Overlay
    current_time_sec = frame_count / fps
    hours = int(current_time_sec // 3600)
    minutes = int((current_time_sec % 3600) // 60)
    seconds = int(current_time_sec % 60)
    timestamp_text = f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}"
    frame_text = f"Frame: {frame_count}"
    cv2.putText(frame_with_contours, timestamp_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_with_contours, frame_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Write frame to output video 
    if out is not None:
        out.write(frame_with_contours)
    cv2.imshow('Frame with Contours', frame_with_contours)
    cv2.imshow('Otsu Thresholded Mask', otsu_thresh)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
print("Video saved as:", output_path)








# Lane Boundary Stabilization Used Techniques:
    # Layer 1 → Rolling window
    # Layer 2 → Convex hull cleanup
    # Layer 3 → Median base outlier removal
    # Layer 4 → Temporal Smoothing Via Lane History (Exponential Moving Average)


    # # --- Robust Vertical centroid connection with lane boundary check ---
    # max_horizontal_deviation = 8.5  # in pixels, allowed horizontal shift between biscuits
    # lane_centroids = {}        
    # lane_boundaries = {}  
    # # Define lane boundaries from column_boundaries
    # for i, (avg_x, (vx, vy, x0, y0)) in enumerate(column_boundaries):
    #     if i == 0:
    #         left_x = 0
    #     else:
    #         left_x = int((column_boundaries[i-1][0] + avg_x) / 2)
    #     if i == len(column_boundaries)-1:
    #         right_x = frame_with_contours.shape[1]
    #     else:
    #         right_x = int((avg_x + column_boundaries[i+1][0]) / 2)
    #     lane_boundaries[i] = (left_x, right_x)
    # # Organize centroids per lane and ensure they are within lane boundary
    # for i, (cx, cy) in enumerate(zip(centroids_x, centroids_y)):
    #     lane_idx = column_labels[i]
    #     left_x, right_x = lane_boundaries[lane_idx]
    #     if left_x <= cx <= right_x:
    #         lane_centroids.setdefault(lane_idx, []).append((cx, cy))
    # # Connect centroids vertically within each lane
    # for lane_idx, points in lane_centroids.items():
    #     if len(points) < 2:
    #         continue
    #     # Sort centroids by vertical position
    #     points_sorted = sorted(points, key=lambda p: p[1])
    #     for i in range(len(points_sorted) - 1):
    #         (cx1, cy1), (cx2, cy2) = points_sorted[i], points_sorted[i+1]
    #         dx = abs(cx2 - cx1)
    #         dy = cy2 - cy1
    #         # Only trigger if horizontal displacement exceeds threshold
    #         if dx > max_horizontal_deviation:
    #             # Mark as misaligned
    #             cv2.circle(frame_with_contours, (cx1, cy1), 5, (0, 0, 255), -1)
    #             cv2.circle(frame_with_contours, (cx2, cy2), 5, (0, 0, 255), -1)
    #             warning_text = f"Misaligned in lane {lane_idx}"
    #             cv2.putText(frame_with_contours, warning_text, (10, 120 + 20 * lane_idx),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #             if frame_count not in misaligned_frames:
    #                 misaligned_frames.add(frame_count)
    #                 print(f"Misalignment detected at frame {frame_count} in lane {lane_idx}")
    #                 image_name = f"misalignment_frame_{frame_count}_lane_{lane_idx}.png"
    #                 cv2.imwrite(os.path.join(misaligned_output_dir, image_name), frame_with_contours)
    #                 winsound.Beep(1800, 400)
    #         # Draw connection line for visualization
    #         cv2.line(frame_with_contours, (cx1, cy1), (cx2, cy2), colors_24[lane_idx], 1)



    # PER LANE ADAPTIVE SLOPE LEVEL INTER LANE STRIP ALLIGNMENT CHECK CODE
    # # --- Robust Vertical centroid connection with lane boundary check (slope-aware) ---
    # max_horizontal_deviation = 8.5  # base allowed horizontal shift between biscuits in pixels
    # lane_centroids = {}
    # lane_boundaries = {}
    # lane_thresholds = []
    # # Step 1: Define lane boundaries from column_boundaries
    # for i, (avg_x, (vx, vy, x0, y0)) in enumerate(column_boundaries):
    #     if i == 0:
    #         left_x = 0
    #     else:
    #         left_x = int((column_boundaries[i-1][0] + avg_x) / 2)
    #     if i == len(column_boundaries)-1:
    #         right_x = frame_with_contours.shape[1]
    #     else:
    #         right_x = int((avg_x + column_boundaries[i+1][0]) / 2)
    #     lane_boundaries[i] = (left_x, right_x)
    #     # Step 2: Compute per-lane slope-based threshold
    #     slope = (vx / vy) if abs(vy) > 1e-3 else 0
    #     margin_factor = 0.5  # tuning constant, px per slope unit
    #     lane_thresholds.append(max_horizontal_deviation + abs(slope) * margin_factor)
    # # Step 3: Organize centroids per lane and ensure they are within lane boundary
    # for i, (cx, cy) in enumerate(zip(centroids_x, centroids_y)):
    #     lane_idx = column_labels[i]
    #     left_x, right_x = lane_boundaries[lane_idx]
    #     if left_x <= cx <= right_x:
    #         lane_centroids.setdefault(lane_idx, []).append((cx, cy))
    # # Step 4: Connect centroids vertically within each lane
    # for lane_idx, points in lane_centroids.items():
    #     if len(points) < 2:
    #         continue
    #     # Sort centroids by vertical position
    #     points_sorted = sorted(points, key=lambda p: p[1])
    #     avg_x, (vx, vy, x0, y0) = column_boundaries[lane_idx]
    #     lane_thresh = lane_thresholds[lane_idx]
    #     # Optional: compute slope from actual centroids for adaptive threshold
    #     centroid_slopes = [(points_sorted[i+1][0]-points_sorted[i][0])/(points_sorted[i+1][1]-points_sorted[i][1]+1e-3)
    #                     for i in range(len(points_sorted)-1)]
    #     if centroid_slopes:
    #         slope_range = max(centroid_slopes) - min(centroid_slopes)
    #         lane_thresh += slope_range * 0.5  # factor can be tuned
    #     # Check each consecutive pair
    #     for i in range(len(points_sorted) - 1):
    #         (cx1, cy1), (cx2, cy2) = points_sorted[i], points_sorted[i+1]
    #         # Project centroids along lane slope
    #         if abs(vy) > 1e-3:
    #             proj_x1 = x0 + (cy1 - y0) * (vx / vy)
    #             proj_x2 = x0 + (cy2 - y0) * (vx / vy)
    #         else:
    #             proj_x1, proj_x2 = cx1, cx2  # vertical lane
    #         dx_along_lane = abs(proj_x2 - proj_x1)
    #         dy = cy2 - cy1
    #         # Only trigger if horizontal deviation along lane exceeds dynamic threshold
    #         if dx_along_lane > lane_thresh:
    #             # Mark as misaligned
    #             cv2.circle(frame_with_contours, (cx1, cy1), 5, (0, 0, 255), -1)
    #             cv2.circle(frame_with_contours, (cx2, cy2), 5, (0, 0, 255), -1)
    #             warning_text = f"Misaligned in lane {lane_idx}"
    #             cv2.putText(frame_with_contours, warning_text, (10, 120 + 20 * lane_idx),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #             if frame_count not in misaligned_frames:
    #                 misaligned_frames.add(frame_count)
    #                 print(f"Misalignment detected at frame {frame_count} in lane {lane_idx}")
    #                 image_name = f"misalignment_frame_{frame_count}_lane_{lane_idx}.png"
    #                 cv2.imwrite(os.path.join(misaligned_output_dir, image_name), frame_with_contours)
    #                 winsound.Beep(1800, 400)
    #         # Draw connection line for visualization
    #         cv2.line(frame_with_contours, (cx1, cy1), (cx2, cy2), colors_24[lane_idx], 1)


    # # ---- PER LANE ADAPTIVE SLOPE LEVEL INTER LANE STRIP ALLIGNMENT CHECK CODE + TEMPORAL FILTER VERSION---- (SO FAR BEST)
    # max_horizontal_deviation = 5     # base allowed deviation
    # margin_factor = 0.01             # slope influence tuning
    # temporal_window = 3              # number of frames to track
    # temporal_trigger_count = 3       # must misalign in N frames to beep
    # lane_centroids = {}
    # lane_boundaries = {}
    # lane_thresholds = {}
    # # Step 1: Define lane boundaries + compute slope-based thresholds
    # for i, (avg_x, (vx, vy, x0, y0)) in enumerate(column_boundaries):
    #     # Compute lane boundaries
    #     if i == 0:
    #         left_x = 0
    #     else:
    #         left_x = int((column_boundaries[i-1][0] + avg_x) / 2)
    #     if i == len(column_boundaries)-1:
    #         right_x = frame_with_contours.shape[1]
    #     else:
    #         right_x = int((avg_x + column_boundaries[i+1][0]) / 2)
    #     lane_boundaries[i] = (left_x, right_x)
    #     slope = (vx / vy) if abs(vy) > 1e-3 else 0
    #     lane_thresholds[i] = max_horizontal_deviation + abs(slope) * margin_factor
    #     if i not in lane_misalignment_history:
    #         lane_misalignment_history[i] = deque(maxlen=temporal_window)
    # # Step 2: Assign centroids to lanes
    # for idx, (cx, cy) in enumerate(zip(centroids_x, centroids_y)):
    #     lane_idx = column_labels[idx]
    #     left_x, right_x = lane_boundaries[lane_idx]
    #     if left_x <= cx <= right_x:
    #         lane_centroids.setdefault(lane_idx, []).append((cx, cy))
    # # Step 3: Check misalignment per lane
    # for lane_idx, points in lane_centroids.items():
    #     bad_count = 0
    #     if len(points) < 2:
    #         lane_misalignment_history[lane_idx].append(False)
    #         continue
    #     points_sorted = sorted(points, key=lambda p: p[1])
    #     avg_x, (vx, vy, x0, y0) = column_boundaries[lane_idx]
    #     lane_thresh = lane_thresholds[lane_idx]
    #     misalignment_detected = False
    #     # Precompute lane normal vector (IMPORTANT FIX)
    #     if abs(vy) > 1e-3:
    #         lane_dx = vx
    #         lane_dy = vy
    #     else:
    #         lane_dx = 0
    #         lane_dy = 1
    #     norm = (lane_dx**2 + lane_dy**2) ** 0.5
    #     lane_dx /= norm
    #     lane_dy /= norm
    #     # Normal direction (perpendicular)
    #     normal_x = -lane_dy
    #     normal_y = lane_dx
    #     for i in range(len(points_sorted) - 1):
    #         (cx1, cy1), (cx2, cy2) = points_sorted[i], points_sorted[i+1]
    #         dx = cx2 - cx1
    #         dy = cy2 - cy1
    #         # Project displacement onto lane normal
    #         dx_along_lane = abs(dx * normal_x + dy * normal_y)
    #         if dx_along_lane > lane_thresh:
    #             bad_count += 1  # misalignment_detected = True   
    #             # Visual markers
    #             cv2.circle(frame_with_contours, (cx1, cy1), 5, (0, 0, 255), -1)
    #             cv2.circle(frame_with_contours, (cx2, cy2), 5, (0, 0, 255), -1)
    #             warning_text = f"Misaligned in lane {lane_idx}"
    #             cv2.putText(frame_with_contours, warning_text,
    #                         (10, 120 + 20 * lane_idx),
    #                         cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.6, (0, 0, 255), 2)
    #         # Visualization line
    #         cv2.line(frame_with_contours,
    #                 (cx1, cy1), (cx2, cy2),
    #                 colors_24[lane_idx], 1)
    #     if bad_count >= 2:
    #         misalignment_detected = True
    #     # Step 4: Rolling Temporal Filter
    #     lane_misalignment_history[lane_idx].append(misalignment_detected)
    #     if lane_misalignment_history[lane_idx].count(True) >= temporal_trigger_count:
    #         if frame_count not in misaligned_frames:
    #             misaligned_frames.add(frame_count)
    #             print(f"Misalignment detected at frame {frame_count} in lane {lane_idx}")
    #             image_name = f"misalignment_frame_{frame_count}_lane_{lane_idx}.png"
    #             cv2.imwrite(os.path.join(misaligned_output_dir, image_name),
    #                         frame_with_contours)
    #             winsound.Beep(1800, 400)