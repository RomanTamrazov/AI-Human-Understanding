import cv2
from detector import PersonDetector
from pose import PoseEstimator
from intent_predictor import IntentPredictor
from emotion import EmotionRecognizer
from hand_gesture import HandGestureRecognizer
from draw import draw_text

# ==================== ИНИЦИАЛИЗАЦИЯ ====================
cap = cv2.VideoCapture(0)

detector = PersonDetector()
pose_estimator = PoseEstimator()
intent_model = IntentPredictor()
emotion_model = EmotionRecognizer()
hand_gesture_model = HandGestureRecognizer()

# ==================== ОКНО FULLSCREEN ====================
WINDOW_NAME = "AI Human Understanding"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# ==================== ОСНОВНОЙ ЦИКЛ ====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame, _ = frame.shape
    boxes = detector.detect(frame)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        person = frame[y1:y2, x1:x2]
        if person.size < 500:
            continue

        # ==================== ПОЗА ====================
        pose = pose_estimator.estimate(person)
        if pose is None:
            continue

        # ==================== ЖЕСТЫ РУК ====================
        gesture_text = "жест: —"
        gesture_prob = 0.0

        try:
            gesture, g_prob = hand_gesture_model.recognize(person)
            if gesture != "нет жеста":
                gesture_text = f"жест: {gesture} ({int(g_prob * 100)}%)"
                gesture_prob = g_prob
        except Exception as e:
            gesture_text = "жест: ошибка"

        # ==================== ЭМОЦИИ ====================
        emotion_text = "эмоция: —"
        face_box = pose_estimator.detect_face(person)

        if face_box:
            fx1, fy1, fx2, fy2 = map(int, face_box)
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(person.shape[1], fx2), min(person.shape[0], fy2)

            face = person[fy1:fy2, fx1:fx2]

            if face.size > 500:
                try:
                    emo, emo_prob = emotion_model.predict(face)
                    emotion_text = f"эмоция: {emo} ({int(emo_prob * 100)}%)"
                except Exception:
                    emotion_text = "эмоция: ошибка"
            else:
                emotion_text = "эмоция: мало лицо"

        # ==================== ДЕЙСТВИЯ + НАМЕРЕНИЕ ====================
        intent_model.update(box, pose)

        action, action_prob = intent_model.detect_action(pose)
        intent, intent_prob = intent_model.detect_intent(pose)

        # ==================== ВЫБОР ГЛАВНОГО ТЕКСТА ====================
        if gesture_prob > action_prob:
            main_text = gesture_text
        else:
            main_text = f"действие: {action} ({int(action_prob * 100)}%)"

        # ==================== ОТРИСОВКА ====================
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frame = draw_text(
            frame,
            main_text,
            (x1, max(30, y1 - 80)),
            34
        )

        frame = draw_text(
            frame,
            f"намерение: {intent} ({int(intent_prob * 100)}%)",
            (x1, max(30, y1 - 45)),
            30
        )

        # ==================== ЭМОЦИЯ (ПРАВЫЙ ВЕРХ БОКСА) ====================
        emotion_x = max(5, x2 - 150)   # 220px — безопасная ширина под текст
        emotion_y = max(5, y1 - 30 )

        frame = draw_text(
            frame,
            emotion_text,
            (emotion_x, emotion_y),
            28
        )


    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
        break

# ==================== ЗАВЕРШЕНИЕ ====================
cap.release()
cv2.destroyAllWindows()
