from ultralytics import YOLO
import cv2
import supervision as sv

MODEL_PATH = 'pretrained-models/yolov8n.pt'
INPUT_PATH = "input/video2.mp4"

LINE_START = sv.Point(850, 240)
LINE_END = sv.Point(0, 240)

# load model
model = YOLO(MODEL_PATH)

line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)

for result in model.track(source=INPUT_PATH, stream=True):

    frame = result.orig_img
    detections = sv.Detections.from_yolov8(result)

    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    detections = detections[detections.class_id == 0] # 0 is class_id for person

    labels = [
        f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]

    frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )

    line_counter.trigger(detections=detections)
    line_annotator.annotate(frame=frame, line_counter=line_counter)

    cv2.imshow("yolov8", frame)

    if (cv2.waitKey(30) == 27):
        break

print("in_count: ")
print(line_counter.in_count)
print("out_count: ")
print(line_counter.out_count)


