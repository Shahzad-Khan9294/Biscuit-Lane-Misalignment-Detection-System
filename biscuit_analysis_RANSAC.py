import cv2
import numpy as np


# Generate 24 unique colors for 24 columns
np.random.seed(42)
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
video_path = 'C:\\Users\\shahzad.khan\\Documents\\Biscuits_Lane_Tracker\\biscuits_vid1.mp4'  # Video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    exit()

# Get video frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Default to 30 fps if not available
frames_for_3_sec = int(fps * 3)

# Skip to this second after computing lines (set to 0 to disable skipping)
skip_to_second = 70

# Define polygon ROI points
polygon_points = np.array([[266, 314], [652, 312], [707, 518], [233, 520]], dtype=np.int32)

# Set up mouse callback
cv2.namedWindow('Frame with Contours')
cv2.setMouseCallback('Frame with Contours', mouse_callback)

# Initialize frame counter
frame_count = 0

# Initialize trajectory image and column boundaries
trajectory_image = None
column_boundaries = None
collecting_trajectories = True

while True:
    # Read frame from video
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Increment frame counter first
    frame_count += 1
    
    # Process only even frames for speed
    if frame_count % 2 != 0:
        continue
    
    # Resize frame to height 600 while maintaining aspect ratio
    height, width = frame.shape[:2]
    
    target_height = 600
    aspect_ratio = width / height
    target_width = int(target_height * aspect_ratio)
    frame = cv2.resize(frame, (target_width, target_height))
    
    # Convert to grayscale for Otsu thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological opening with 3x3 kernel
    kernel = np.ones((3, 3), np.uint8)
    otsu_thresh = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
    
    # Create polygon mask
    mask = np.zeros_like(otsu_thresh)
    cv2.fillPoly(mask, [polygon_points], 255)
    
    # Apply mask to thresholded image to get ROI
    roi_thresh = cv2.bitwise_and(otsu_thresh, mask)
    
    # Detect contours in the masked region
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours with width < 30 and height < 30, and minimum width/height of 3
    filtered_contours = []
    widths = []
    heights = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 3 and h >= 3 and w < 100 and h < 100:
            filtered_contours.append(contour)
            widths.append(w)
            heights.append(h)
    
    # Draw contours on the RGB frame
    frame_with_contours = frame.copy()
    
    # Initialize trajectory image on first frame
    if trajectory_image is None:
        trajectory_image = np.zeros((target_height, target_width), dtype=np.uint8)
    
    # Collect trajectories for first 3 seconds
    if collecting_trajectories and frame_count < frames_for_3_sec:
        # Draw x2 positions on trajectory image
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x2 = x + w
            y_center = y + h // 2
            # Draw a point at (x2, y_center)
            cv2.circle(trajectory_image, (x2, y_center), 1, 255, -1)
    
    # After 3 seconds, detect lines and compute column boundaries
    elif collecting_trajectories and frame_count == frames_for_3_sec:
        collecting_trajectories = False
        
        # Display trajectory image
        cv2.imshow('Trajectory Image', trajectory_image)
        cv2.waitKey(1000)  # Display for 1 second
        
        # Find contours of trajectory lines
        contours_traj, _ = cv2.findContours(trajectory_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours_traj is not None and len(contours_traj) > 0:
            # Fit lines to contours and store their properties
            fitted_lines = []
            
            for contour in contours_traj:
                # Only process contours with sufficient points
                if len(contour) >= 10:
                    # Fit a line to the contour points
                    [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    
                    # Calculate contour length (as a measure of significance)
                    contour_length = cv2.arcLength(contour, False)
                    
                    # Get average x position of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        avg_x = M["m10"] / M["m00"]
                    else:
                        avg_x = x0[0]
                    
                    # Store line info: (avg_x, contour_length, line_params)
                    fitted_lines.append((avg_x, contour_length, (vx[0], vy[0], x0[0], y0[0])))
            
            # Sort by contour length (descending) to prioritize longer trajectories
            fitted_lines.sort(key=lambda x: x[1], reverse=True)
            
            # Select non-overlapping lines with minimum x-axis separation
            min_x_distance = 10  # Minimum pixels between lines
            selected_lines = []
            
            for line_info in fitted_lines:
                avg_x, length, line_params = line_info
                # Check if this line is far enough from all already selected lines
                is_far_enough = True
                for selected_x, _, _ in selected_lines:
                    if abs(avg_x - selected_x) < min_x_distance:
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    selected_lines.append(line_info)
                    
                # Stop if we have 24 lines
                if len(selected_lines) >= 24:
                    break
            
            # Sort by x position to get column boundaries from left to right
            selected_lines.sort(key=lambda x: x[0])
            
            # Create visualization RGB image and draw fitted lines in green
            visualization_image = cv2.cvtColor(trajectory_image, cv2.COLOR_GRAY2BGR)
            
            for avg_x, length, (vx, vy, x0, y0) in selected_lines:
                # Calculate line endpoints for drawing (extend across image height)
                # Line equation: (x,y) = (x0,y0) + t*(vx,vy)
                # For y=0: t = -y0/vy, x = x0 - y0*vx/vy
                # For y=height: t = (height-y0)/vy, x = x0 + (height-y0)*vx/vy
                if abs(vy) > 0.001:  # Avoid division by zero
                    t1 = -y0 / vy
                    x1 = int(x0 + t1 * vx)
                    y1 = 0
                    
                    t2 = (target_height - y0) / vy
                    x2 = int(x0 + t2 * vx)
                    y2 = target_height
                    
                    cv2.line(visualization_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display visualization
            cv2.imshow('Fitted Lines Visualization', visualization_image)
            cv2.waitKey(2000)  # Display for 2 seconds
            
            # Store column boundaries as line parameters (not just x positions)
            column_boundaries = [(avg_x, (vx, vy, x0, y0)) for avg_x, _, (vx, vy, x0, y0) in selected_lines]
        else:
            # Fallback: if no lines detected, use empty list
            column_boundaries = []
        
        # Skip to specified second if enabled
        if skip_to_second > 0:
            skip_to_frame = int(skip_to_second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_to_frame)
            frame_count = skip_to_frame
            print(f"Skipped to {skip_to_second} seconds (frame {skip_to_frame})")
    
    # Calculate centroids of all filtered contours
    centroids_x = []
    centroids_y = []
    for contour in filtered_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids_x.append(cx)
            centroids_y.append(cy)
        else:
            x, y, w, h = cv2.boundingRect(contour)
            centroids_x.append(x + w // 2)
            centroids_y.append(y + h // 2)
    
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
    
    # Draw white tilted lines at column boundaries
    if column_boundaries is not None and len(column_boundaries) > 0:
        for avg_x, (vx, vy, x0, y0) in column_boundaries:
            # Draw tilted line from top to bottom
            if abs(vy) > 0.001:
                t1 = -y0 / vy
                x1 = int(x0 + t1 * vx)
                y1 = 0
                
                t2 = (target_height - y0) / vy
                x2 = int(x0 + t2 * vx)
                y2 = target_height
                
                cv2.line(frame_with_contours, (x1, y1), (x2, y2), (255, 255, 255), 1)
    
    # Assign contours to columns using tilted boundaries
    if column_boundaries is not None and len(column_boundaries) > 0 and len(filtered_contours) > 0:
        # Initialize column labels array
        column_labels = []
        
        # Assign each contour based on which tilted boundary region centroid falls in
        for i in range(len(centroids_x)):
            cx = centroids_x[i]
            cy = centroids_y[i]
            
            # Find which column the centroid belongs to
            col = 0
            for j, (avg_x, (vx, vy, x0, y0)) in enumerate(column_boundaries):
                # Calculate x position of this line at the centroid's y coordinate
                if abs(vy) > 0.001:
                    line_x_at_cy = x0 + (cy - y0) * (vx / vy)
                else:
                    line_x_at_cy = x0
                
                # If centroid is to the left of this line, assign to this column
                if cx <= line_x_at_cy:
                    col = j
                    break
            else:
                # If centroid is beyond all boundaries, assign to last column
                col = min(23, len(column_boundaries) - 1)
            
            column_labels.append(col)
        
        # Draw each contour with its cluster color
        for i, contour in enumerate(filtered_contours):
            cluster_label = min(column_labels[i], 23)  # Ensure within color range
            color = colors_24[cluster_label]
            cv2.drawContours(frame_with_contours, [contour], -1, color, 1)
        
        # Group contours by column and draw polylines
        column_contours = {col: [] for col in range(24)}
        for i, contour in enumerate(filtered_contours):
            cluster_label = min(column_labels[i], 23)  # Ensure within color range
            cx = centroids_x[i]
            cy = centroids_y[i]
            column_contours[cluster_label].append((cy, cx))  # Store as (y, x) for sorting by y
        
        # Draw polylines for each column and calculate max gaps
        for col in range(24):
            if len(column_contours[col]) > 1:
                # Sort by Y coordinate (top to bottom)
                sorted_points = sorted(column_contours[col], key=lambda p: p[0])
                
                # Calculate max y distance between consecutive contours
                max_y_gap = 0
                for i in range(len(sorted_points) - 1):
                    y_gap = sorted_points[i + 1][0] - sorted_points[i][0]
                    max_y_gap = max(max_y_gap, y_gap)
                
                # Convert to (x, y) for drawing
                polyline_points = np.array([(x, y) for y, x in sorted_points], dtype=np.int32)
                # Draw polyline with column color
                cv2.polylines(frame_with_contours, [polyline_points], isClosed=False, 
                            color=colors_24[col], thickness=2)
                
                # Display max gap on top of image at this column's x position
                # Get average x position for this column
                avg_column_x = int(np.mean([x for _, x in sorted_points]))
                text = f"{int(max_y_gap)}"
                cv2.putText(frame_with_contours, text, (avg_column_x - 10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors_24[col], 1)
    else:
        # If boundaries not yet computed or no contours, draw them all in green
        cv2.drawContours(frame_with_contours, filtered_contours, -1, (0, 255, 0), 1)
    
    # Draw red hollow polygon ROI
    cv2.polylines(frame_with_contours, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=1)
    
    # Display time at top left
    timestamp_seconds = frame_count / fps
    time_text = f"Time: {timestamp_seconds:.2f}s"
    cv2.putText(frame_with_contours, time_text, (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display frame number at top right
    frame_text = f"Frame: {frame_count}"
    text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = target_width - text_size[0] - 10
    cv2.putText(frame_with_contours, frame_text, (text_x, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display frame with contours and thresholded mask
    cv2.imshow('Frame with Contours', frame_with_contours)
    cv2.imshow('Otsu Thresholded Mask', otsu_thresh)
    
    # Wait for key press (30ms delay for smooth video playback)
    key = cv2.waitKey(30) & 0xFF
    
    # Quit if 'q' is pressed
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


