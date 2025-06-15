import cv2
import torch
import numpy as np

# Load YOLOv5m model (detect only cars)
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.conf = 0.4
model.classes = [2]  # Only detect cars

# Open input video
video_path = 'b2.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_brake_lights.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    zone_top = h // 2
    zone_bottom = h
    zone_left = int(w * 0.1)
    zone_right = int(w * 0.9)

    results = model(frame)
    cars = results.xyxy[0].cpu().numpy()

    for car in cars:
        x1, y1, x2, y2, conf, cls = map(int, car[:6])
        car_area = (x2 - x1) * (y2 - y1)

        inter_left = max(x1, zone_left)
        inter_right = min(x2, zone_right)
        inter_top = max(y1, zone_top)
        inter_bottom = min(y2, zone_bottom)
        inter_area = max(0, inter_right - inter_left) * max(0, inter_bottom - inter_top)

        if inter_area / car_area < 0.6:
            continue

        height_car = y2 - y1
        shrink_y1 = y1 + int(0.3 * height_car)
        shrink_y2 = y2 - int(0.3 * height_car)

        roi = frame[shrink_y1:shrink_y2, x1:x2]
        if roi.size == 0:
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 60, 40])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([160, 60, 40])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        brake_light_count = 0
        best_threshold = None

        for threshold in range(130, 201):  # 145 to 200 inclusive
            v = hsv[:, :, 2]
            bright_mask = cv2.inRange(v, threshold, 255)
            final_mask = cv2.bitwise_and(red_mask, bright_mask)
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 30:
                    continue
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x1 + x, shrink_y1 + y), (x1 + x + w_box, shrink_y1 + y + h_box), (0, 255, 0), 2)
                count += 1

            if count > 0:
                brake_light_count = count
                best_threshold = threshold
                break  # Found brake lights at this threshold, stop checking higher ones

        cv2.rectangle(frame, (x1, shrink_y1), (x2, shrink_y2), (255, 0, 0), 2)
        cv2.putText(frame, f'Brake Lights: {brake_light_count}', (x1, shrink_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if best_threshold:
            cv2.putText(frame, f'Bright T: {best_threshold}', (x1, shrink_y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    out.write(frame)
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
