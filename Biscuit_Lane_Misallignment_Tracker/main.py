import cv2
import winsound
import numpy as np
from config import *
from frame_processing import *


# Mouse callback
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates: x={x}, y={y}")

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video.")
        exit()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frames_for_3_sec = int(fps * 20)
    skip_to_second = 70

    # ROI polygon
    polygon_points = np.array([[266, 314], [652 - 4.5, 312], [707 - 4.5, 518], [233, 520]], dtype=np.int32)

    # Setup window and mouse callback
    cv2.namedWindow('Frame with Contours')
    cv2.setMouseCallback('Frame with Contours', mouse_callback)

    frame_count = 0
    trajectory_image = None
    column_boundaries = None
    collecting_trajectories = True
    misaligned_frames = set()
    trajectory_queue = []
    lane_history = [[] for _ in range(EXPECTED_LANES)]

    # Processing Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 2 != 0:
            continue
        frame, target_width, target_height = resize_frame(frame)
        if out is None:
            out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps / 2, (target_width, target_height))

        otsu_thresh = threshold_frame(frame)
        roi_thresh = apply_roi_mask(otsu_thresh, polygon_points)
        filtered_contours = get_filtered_contours(roi_thresh)
        frame_with_contours, shift_left, shift_right, top_left, top_right, bottom_left, bottom_right = draw_polygon_lines(frame.copy(), polygon_points)

        # Initialize trajectory image
        if trajectory_image is None:
            trajectory_image = np.zeros((target_height, target_width), dtype=np.uint8)

        # Initial trajectories collection
        if collecting_trajectories and frame_count < frames_for_3_sec:
            trajectory_image = collect_initial_trajectories(filtered_contours, trajectory_image)

        elif collecting_trajectories and frame_count == frames_for_3_sec:
            collecting_trajectories = False
            cv2.imshow('Trajectory Image', trajectory_image)
            cv2.waitKey(1000)

            column_boundaries = fit_lines_from_trajectories(trajectory_image)

            if skip_to_second > 0:
                skip_to_frame = int(skip_to_second * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, skip_to_frame)
                frame_count = skip_to_frame
                print(f"Skipped to {skip_to_second} seconds (frame {skip_to_frame})")

        # Update trajectory queue and lanes
        trajectory_queue = update_trajectory_queue(trajectory_queue, filtered_contours)
        trajectory_image = rebuild_trajectory_image(trajectory_queue, target_height, target_width)
        lane_history, column_boundaries = update_lane_history_and_columns(lane_history, EXPECTED_LANES, trajectory_image)

        # Compute centroids and assign columns
        centroids_x, centroids_y = compute_centroids(filtered_contours)
        column_labels, misaligned_lanes = assign_centroids_to_columns(centroids_x, centroids_y, column_boundaries)

        # Check extreme boundary crossings
        left_x1, left_y1 = int(top_left[0] + shift_left), int(top_left[1])
        left_x2, left_y2 = int(bottom_left[0] + shift_left), int(bottom_left[1])
        right_x1, right_y1 = int(top_right[0] - shift_right), int(top_right[1])
        right_x2, right_y2 = int(bottom_right[0] - shift_right), int(bottom_right[1])
        extreme_crossed = check_extreme_boundary_crossing(centroids_x, centroids_y, column_labels, left_x1, left_x2, left_y1, left_y2, right_x1, right_x2, right_y1, right_y2)

        # Overlay timestamp & frame number
        frame_with_contours = overlay_timestamps(frame_with_contours, frame_count, fps)
        # Write output
        write_frame_to_output(out, frame_with_contours)
        # Display frames
        cv2.imshow('Frame with Contours', frame_with_contours)
        cv2.imshow('Otsu Thresholded Mask', otsu_thresh)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("Video saved as:", OUTPUT_PATH)


if __name__ == "__main__":
    main()