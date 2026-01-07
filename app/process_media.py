import cv2
from detector import PersonDetector
from pose import PoseEstimator
from intent_predictor import IntentPredictor
from emotion import EmotionRecognizer
from draw import draw_text

detector = PersonDetector()
pose_estimator = PoseEstimator()
intent_model = IntentPredictor()
emotion_model = EmotionRecognizer()

def process_frame(frame):
    h, w, _ = frame.shape
    boxes = detector.detect(frame)

    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        person = frame[y1:y2, x1:x2]
        if person.size == 0:
            continue

        pose = pose_estimator.estimate(person)
        if pose is None:
            continue

        # -------- эмоции --------
        emotion_text = "эмоция: —"
        face_box = pose_estimator.detect_face(person)
        if face_box:
            fx1, fy1, fx2, fy2 = [int(c) for c in face_box]
            face = person[fy1:fy2, fx1:fx2]
            if face.size > 400:
                emo, emo_prob = emotion_model.predict(face)
                emotion_text = f"эмоция: {emo} ({int(emo_prob*100)}%)"

        # -------- действия --------
        intent_model.update(box, pose)
        action, action_prob = intent_model.detect_action(pose)
        intent, intent_prob = intent_model.detect_intent(pose)

        # -------- отрисовка --------
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        frame = draw_text(
            frame,
            f"действие: {action} ({int(action_prob*100)}%)",
            (x1, max(20, y1-60)),
            28
        )
        frame = draw_text(
            frame,
            f"намерение: {intent}",
            (x1, max(20, y1-30)),
            24
        )
        frame = draw_text(
            frame,
            emotion_text,
            (x1, min(h-20, y2+25)),
            24
        )

    return frame


def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    out = process_frame(img)
    cv2.imwrite(output_path, out)


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        out.write(frame)

    cap.release()
    out.release()
