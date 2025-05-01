# utils.py (UPDATED with full OpenCV lane + polygon logic)

import cv2
import numpy as np
import os
import torch
import subprocess
from sort import Sort

def forecast_next_n_steps(model, history, steps=10):
    model.eval()
    preds = []
    input_seq = history.copy()
    for _ in range(steps):
        inp = torch.tensor([input_seq], dtype=torch.float32)
        with torch.no_grad():
            pred = model(inp)
        pred_np = pred.cpu().numpy()[0]
        preds.append(pred_np)
        input_seq.append(pred_np.tolist())
        input_seq.pop(0)
    return np.array(preds)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(img, mask)

def average_line(lines):
    if len(lines) == 0:
        return None
    x_coords = []
    y_coords = []
    for x1, y1, x2, y2 in lines:
        x_coords += [x1, x2]
        y_coords += [y1, y2]
    return np.polyfit(y_coords, x_coords, deg=1)

def calculate_speed(track_id, cx, cy, fps, frame_height, previous_positions):
    if track_id in previous_positions:
        prev_cx, prev_cy = previous_positions[track_id]
        pixel_distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
        scale = (frame_height - cy) / frame_height
        meters_per_pixel = 0.02 + (scale * 0.08)
        distance_moved_meters = pixel_distance * meters_per_pixel
        speed = (distance_moved_meters * fps) * 3.6
    else:
        speed = 0
    previous_positions[track_id] = (cx, cy)
    return round(speed, 2)

def process_video(video_path, yolo_model, lstm_model):
    tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)
    previous_positions = {}
    vehicle_directions = {}
    traffic_history = []
    forecast_plot_data = []
    forecast_plot_timestamps = []

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the first frame from video!")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_interval = int(fps)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = "static/processed/output.avi"
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Lane detection on first frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    roi_vertices = np.array([[
        (0, frame_height),
        (frame_width//2 - 80, frame_height//2 + 50),
        (frame_width//2 + 80, frame_height//2 + 50),
        (frame_width, frame_height)
    ]], dtype=np.int32)
    edges_roi = region_of_interest(edges, roi_vertices)
    lines = cv2.HoughLinesP(edges_roi, 1, np.pi/180, threshold=60, minLineLength=50, maxLineGap=200)
    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.3 and x1 < frame_width//2 and x2 < frame_width//2:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3 and x1 > frame_width//2 and x2 > frame_width//2:
                right_lines.append((x1, y1, x2, y2))
    left_fit = average_line(left_lines)
    right_fit = average_line(right_lines)

    if left_fit is not None:
        left_top = int(left_fit[0] * (frame_height//2) + left_fit[1])
        left_bottom = int(left_fit[0] * frame_height + left_fit[1])
        vertices1 = np.array([[0, frame_height//2], [left_top, frame_height//2],
                              [left_bottom, frame_height], [0, frame_height]], dtype=np.int32)
    else:
        vertices1 = np.array([[0, frame_height//2]]*2 + [[0, frame_height]]*2, dtype=np.int32)

    if right_fit is not None:
        right_top = int(right_fit[0] * (frame_height//2) + right_fit[1])
        right_bottom = int(right_fit[0] * frame_height + right_fit[1])
        vertices2 = np.array([[right_top, frame_height//2], [frame_width, frame_height//2],
                              [frame_width, frame_height], [right_bottom, frame_height]], dtype=np.int32)
    else:
        vertices2 = np.array([[frame_width, frame_height//2]]*2 + [[frame_width, frame_height]]*2, dtype=np.int32)

    cap.release()
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    orange_color = (0, 165, 255)
    heavy_traffic_threshold = 10

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, conf=0.5)
        processed_frame = results[0].plot(line_width=1)
        cv2.polylines(processed_frame, [vertices1], True, (0, 255, 0), 2)
        cv2.polylines(processed_frame, [vertices2], True, (255, 0, 0), 2)

        detections = []
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, conf.item()])
        detections = np.array(detections) if detections else np.empty((0, 5))
        tracked = tracker.update(detections)

        vehicles_left = 0
        vehicles_right = 0

        for x1, y1, x2, y2, track_id in tracked:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if track_id in vehicle_directions:
                prev_x, _ = vehicle_directions[track_id]
                if cx > prev_x:
                    cv2.putText(processed_frame, "Wrong Direction!", (int(x1), int(y1)-50), font, 0.7, (0,0,255), 2)
            vehicle_directions[track_id] = (cx, cy)

            speed = calculate_speed(track_id, cx, cy, fps, frame_height, previous_positions)

            color = (0, 255, 0) if speed <= 90 else (0, 0, 255)
            if speed > 90:
                cv2.putText(processed_frame, "Slow Down!", (int(x1), int(y1)-30), font, 0.7, (0, 0, 255), 2)

            if x1 < frame_width // 2:
                vehicles_left += 1
            else:
                vehicles_right += 1

            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(processed_frame, f"ID {int(track_id)}", (int(x1), int(y1)-10), font, 0.7, color, 2)
            cv2.putText(processed_frame, f"Speed: {speed} km/h", (int(x1), int(y2)+20), font, 0.7, color, 2)

        frame_count += 1
        if frame_count % fps_interval == 0:
            traffic_history.append([vehicles_left, vehicles_right])
            if len(traffic_history) > 10:
                traffic_history.pop(0)

            if len(traffic_history) == 10:
                future_preds = forecast_next_n_steps(lstm_model, traffic_history.copy(), steps=10)
                left_10s_pred, right_10s_pred = future_preds[9]
                last_forecast_frame_index = frame_count
                forecast_10s_text = (int(left_10s_pred), int(right_10s_pred))

        if 'last_forecast_frame_index' in locals() and frame_count - last_forecast_frame_index <= int(fps * 5):
            if forecast_10s_text:
                cv2.putText(processed_frame, f"+10s Left: {forecast_10s_text[0]}", (10, 200), font, 0.7, (255, 255, 0), 2)
                cv2.putText(processed_frame, f"+10s Right: {forecast_10s_text[1]}", (820, 200), font, 0.7, (255, 255, 0), 2)

        cv2.putText(processed_frame, f"Vehicles in Left Lane: {vehicles_left}", (10, 50), font, font_scale, orange_color, 2)
        cv2.putText(processed_frame, f"Traffic Intensity: {'Heavy' if vehicles_left > heavy_traffic_threshold else 'Smooth'}", (10, 100), font, font_scale, orange_color, 2)
        cv2.putText(processed_frame, f"Vehicles in Right Lane: {vehicles_right}", (820, 50), font, font_scale, orange_color, 2)
        cv2.putText(processed_frame, f"Traffic Intensity: {'Heavy' if vehicles_right > heavy_traffic_threshold else 'Smooth'}", (820, 100), font, font_scale, orange_color, 2)

        out.write(processed_frame)

    cap.release()
    out.release()

    mp4_path = output_path.replace('.avi', '.mp4')
    ffmpeg_exe = "ffmpeg.exe"
    result = subprocess.run(
        [ffmpeg_exe, "-y", "-loglevel", "error", "-i", output_path, mp4_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("❌ FFmpeg error:", result.stderr)
        raise RuntimeError("MP4 conversion failed.")
    return mp4_path
