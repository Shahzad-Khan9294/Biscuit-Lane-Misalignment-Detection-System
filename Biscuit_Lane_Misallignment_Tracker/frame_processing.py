import cv2
import winsound
import numpy as np
from config import *


# Frame Utilities
def resize_frame(frame):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    target_width = int(TARGET_HEIGHT * aspect_ratio)
    frame = cv2.resize(frame, (target_width, TARGET_HEIGHT))
    return frame, target_width, TARGET_HEIGHT

def threshold_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    otsu_thresh = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
    return otsu_thresh

def apply_roi_mask(thresh, polygon_points):
    mask = np.zeros_like(thresh)
    cv2.fillPoly(mask, [polygon_points], 255)
    roi_thresh = cv2.bitwise_and(thresh, mask)
    return roi_thresh

def get_filtered_contours(roi_thresh):
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 3 and h >= 3 and w < 100 and h < 100:
            filtered_contours.append(contour)
    return filtered_contours

def draw_polygon_lines(frame, polygon_points):
    cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 255, 255), thickness=1)
    shift_right = 7
    shift_left = 5
    top_left, top_right, bottom_right, bottom_left = polygon_points
    cv2.line(frame,
             (int(top_left[0] + shift_left), int(top_left[1])),
             (int(bottom_left[0] + shift_left), int(bottom_left[1])),
             (0, 0, 0), 1)
    cv2.line(frame,
             (int(top_right[0] - shift_right), int(top_right[1])),
             (int(bottom_right[0] - shift_right), int(bottom_right[1])),
             (0, 0, 0), 1)
    return frame, shift_left, shift_right, top_left, top_right, bottom_left, bottom_right

def compute_centroids(filtered_contours):
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
    return centroids_x, centroids_y

# Trajectory & Lane Functions
def collect_initial_trajectories(filtered_contours, trajectory_image):
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        x2 = x + w
        y_center = y + h // 2
        cv2.circle(trajectory_image, (x2, y_center), 1, 255, -1)
    return trajectory_image

def fit_lines_from_trajectories(trajectory_image):
    contours_traj, _ = cv2.findContours(trajectory_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fitted_lines = []
    if contours_traj:
        for contour in contours_traj:
            if len(contour) >= 10:
                vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
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
    return column_boundaries

def update_trajectory_queue(trajectory_queue, filtered_contours):
    current_frame_points = [(x + w, y + h // 2) for x, y, w, h in [cv2.boundingRect(c) for c in filtered_contours]]
    trajectory_queue.append(current_frame_points)
    if len(trajectory_queue) > ROLLING_WINDOW_SIZE:
        trajectory_queue.pop(0)
    return trajectory_queue

def rebuild_trajectory_image(trajectory_queue, target_height, target_width):
    trajectory_image = np.zeros((target_height, target_width), dtype=np.uint8)
    for traj_frame in trajectory_queue:
        for x2, y_center in traj_frame:
            cv2.circle(trajectory_image, (x2, y_center), 1, 255, -1)
    return trajectory_image

def update_lane_history_and_columns(lane_history, EXPECTED_LANES, trajectory_image):
    contours_traj, _ = cv2.findContours(trajectory_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fitted_lines = []
    if contours_traj:
        for contour in contours_traj:
            if len(contour) >= 10:
                vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
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

    column_boundaries = []
    smoothed_lanes = [None] * EXPECTED_LANES
    for i in range(EXPECTED_LANES):
        if i < len(selected_lines):
            _, _, line_params = selected_lines[i]
            lane_history[i].append(line_params)
            if len(lane_history[i]) > MAX_HISTORY_LENGTH:
                lane_history[i].pop(0)
        if lane_history[i]:
            vx, vy, x0, y0 = np.median(lane_history[i], axis=0)
            if smoothed_lanes[i] is None:
                smoothed_lanes[i] = np.array([vx, vy, x0, y0])
            else:
                smoothed_lanes[i] = (ALPHA * np.array([vx, vy, x0, y0]) + (1 - ALPHA) * smoothed_lanes[i])
            vx_s, vy_s, x0_s, y0_s = smoothed_lanes[i]
            column_boundaries.append((x0_s, (vx_s, vy_s, x0_s, y0_s)))
        else:
            column_boundaries.append((0, (0,1,0,0)))
    return lane_history, column_boundaries

def assign_centroids_to_columns(centroids_x, centroids_y, column_boundaries):
    column_labels = []
    misaligned_lanes = {}
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
        if diff > CROSS_THRESHOLD:
            if col not in misaligned_lanes:
                misaligned_lanes[col] = []
            misaligned_lanes[col].append((cx, cy))
    return column_labels, misaligned_lanes

def check_extreme_boundary_crossing(centroids_x, centroids_y, column_labels, left_x1, left_x2, left_y1, left_y2,
                                    right_x1, right_x2, right_y1, right_y2):
    extreme_crossed = {"left": [], "right": []}
    for i, (cx, cy) in enumerate(zip(centroids_x, centroids_y)):
        lane_idx = column_labels[i]
        if lane_idx == 0:
            left_line_x = left_x1 + (cx - left_x1) * 0 if abs(left_y2 - left_y1) < 1e-3 else left_x1 + (cy - left_y1) / (left_y2 - left_y1) * (left_x2 - left_x1)
            if cx < left_line_x - EXTREME_THRESHOLD:
                extreme_crossed["left"].append((cx, cy))
        if lane_idx == EXPECTED_LANES-1:
            right_line_x = right_x1 + (cx - right_x1) * 0 if abs(right_y2 - right_y1) < 1e-3 else right_x1 + (cy - right_y1) / (right_y2 - right_y1) * (right_x2 - right_x1)
            if cx > right_line_x + EXTREME_THRESHOLD:
                extreme_crossed["right"].append((cx, cy))
    return extreme_crossed

def overlay_timestamps(frame_with_contours, frame_count, fps):
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
    return frame_with_contours

def write_frame_to_output(out, frame_with_contours):
    if out is not None:
        out.write(frame_with_contours)