import cv2
import time
from ultralytics import YOLO
from app.weight_estimator import estimate_weight_index

BIRD_CLASS_ID = 14  # COCO class for bird

# -------------------------------
# Motion fallback (birds only)
# -------------------------------
def motion_fallback(frame, min_area=400):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    if not hasattr(motion_fallback, "bg"):
        motion_fallback.bg = blur
        return []

    diff = cv2.absdiff(motion_fallback.bg, blur)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_fallback.bg = blur

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))

    return boxes


# -------------------------------
# Main processing
# -------------------------------
def process_video(video_path, output_path):

    model = YOLO("yolov8n.pt",verbose=False)

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(fps),
        (width, height)
    )

    unique_ids = set()
    time_series = []

    frame_no = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            conf=0.01,
            iou=0.25,
            imgsz=1280,
            max_det=3000
        )

        active_birds = 0
        weight_sum = 0
        drawn = False

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            ids = results[0].boxes.id
            ids = ids.cpu().numpy() if ids is not None else [None]*len(boxes)

            for box, cls_id, tid in zip(boxes, classes, ids):

                # 🚨 FILTER ONLY BIRDS
                if int(cls_id) != BIRD_CLASS_ID:
                    continue

                drawn = True
                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1

                active_birds += 1
                wt = estimate_weight_index(w, h)
                weight_sum += wt

                label = f"WtIdx:{wt}"
                if tid is not None:
                    unique_ids.add(int(tid))
                    label = f"ID:{int(tid)} WtIdx:{wt}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0,255,0),
                    1
                )

        # ---------------- Fallback if YOLO finds nothing ----------------
        if not drawn:
            boxes = motion_fallback(frame)
            for (x1, y1, x2, y2) in boxes:
                active_birds += 1
                wt = estimate_weight_index(x2-x1, y2-y1)
                weight_sum += wt
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)

        avg_weight = round(weight_sum / active_birds, 2) if active_birds else 0
        timestamp = round(frame_no / fps, 2)

        time_series.append({
            "timestamp_sec": timestamp,
            "active_birds": active_birds,
            "average_weight_index": avg_weight
        })

        cv2.putText(
            frame,
            f"Active Birds: {active_birds}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        out.write(frame)
        frame_no += 1

    cap.release()
    out.release()

    processing_fps = round(frame_no / (time.time() - start_time), 2)

    return {
        "total_unique_birds": len(unique_ids),
        "processing_fps": processing_fps,
        "counts": time_series,
        "weight_estimates": "relative_weight_index",
        "artifacts": {
            "annotated_video": output_path
        }
    }
