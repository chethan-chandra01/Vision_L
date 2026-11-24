import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load CPU-optimized OpenVINO model
model = YOLO("../models/best_openvino_model/")
cap = cv2.VideoCapture("test_video.mp4")

# Assumed camera intrinsics (standard 640x480 webcam/Gazebo camera)
FOCAL_LENGTH_PX = 550.0  # f_x and f_y in pixels
ASSUMED_OBJ_HEIGHT_M = 0.5  # Average obstacle height in meters (for monocular fallback)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    img_center_x = w // 2

    # Draw robot's central navigation corridor (danger zone)
    corridor_left = int(w * 0.35)
    corridor_right = int(w * 0.65)
    cv2.line(frame, (corridor_left, 0), (corridor_left, h), (255, 255, 0), 1)
    cv2.line(frame, (corridor_right, 0), (corridor_right, h), (255, 255, 0), 1)

    # 2. Run Detection + ByteTrack
    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", device="cpu", verbose=False)

    nav_command = "STATUS: CLEAR - MOVE FORWARD"
    nav_color = (0, 255, 0)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            box_h = max(y2 - y1, 1)

            # 3. Estimate Spatial Coordinates (X, Z) in meters
            # (In Level 5 Gazebo, replace Z with: depth_map[cy, cx])
            Z = (ASSUMED_OBJ_HEIGHT_M * FOCAL_LENGTH_PX) / box_h
            X = ((cx - img_center_x) * Z) / FOCAL_LENGTH_PX

            # 4. Determine Obstacle Risk & Color
            in_corridor = corridor_left < cx < corridor_right
            if Z < 1.2 and in_corridor:
                color = (0, 0, 255)  # Red: Immediate obstacle
                steer = "RIGHT" if X < 0 else "LEFT"
                nav_command = f"ALERT: OBSTACLE ({Z:.1f}m) -> STEER {steer}"
                nav_color = (0, 0, 255)
            elif Z < 2.5 and in_corridor:
                color = (0, 165, 255)  # Orange: Caution
                if nav_color != (0, 0, 255):
                    nav_command = f"CAUTION: OBJECT AHEAD ({Z:.1f}m) -> SLOW DOWN"
                    nav_color = (0, 165, 255)
            else:
                color = (0, 255, 0)  # Green: Safe

            # Draw box, center point, and 3D spatial telemetry
            label = f"ID:{track_id} {model.names[cls_id]} | Z:{Z:.1f}m X:{X:+.1f}m"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display navigation decision banner at the top
    cv2.rectangle(frame, (0, 0), (w, 35), (30, 30, 30), -1)
    cv2.putText(frame, nav_command, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, nav_color, 2)

    cv2.imshow("Percepta - Level 4 Spatial Perception", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
