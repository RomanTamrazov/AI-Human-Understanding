import cv2
import numpy as np
from deepface import DeepFace  # Современная библиотека для распознавания эмоций (лучше FER+)

# Перевод эмоций из DeepFace в русский (сопоставление по вашему списку)
EMOTION_MAP = {
    "angry": "злой",
    "disgust": "отвращение",
    "fear": "испуган",
    "happy": "счастлив",
    "sad": "грустный",
    "surprise": "удивлён",
    "neutral": "нейтральная"
}

class EmotionRecognizer:
    def __init__(self):
        # DeepFace не требует сессии ONNX — она использует встроенные модели (VGG-Face и др.)
        # Если нужно fallback на ONNX, можно добавить, но DeepFace лучше по умолчанию
        self.use_deepface = True  # Флаг для переключения (если ONNX нужен — False)
        try:
            import onnxruntime as ort
            import scipy.special
            self.session = ort.InferenceSession(
                "models/emotion/emotion-ferplus.onnx",
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.softmax = scipy.special.softmax  # Для ONNX
            self.emotions_ru = [  # EMOTIONS_RU для ONNX (из вашего оригинала)
                "нейтральная", "счастлив", "удивлён", "грустный",
                "злой", "испуган", "отвращение", "презрение"
            ]
        except Exception as e:
            print(f"Предупреждение: ONNX не загружен ({e}). Используем только DeepFace.")
            self.use_deepface = True

    def predict(self, face_img):
        """
        Предсказывает эмоцию на изображении лица.
        Использует DeepFace по умолчанию (лучшая точность, меньше bias к neutral).
        Возвращает: (эмоция_ру, вероятность [0-1])
        """
        if self.use_deepface:
            try:
                # DeepFace анализ (enforce_detection=False, т.к. лицо уже обрезано)
                result = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',  # Быстрый бэкенд
                    silent=True  # Без лишних логов
                )
                emo_en = result[0]['dominant_emotion']
                emo_ru = EMOTION_MAP.get(emo_en, emo_en)  # Перевод или fallback
                emo_prob = result[0]['emotion'][emo_en] / 100.0  # [0-1]
                # Порог: если вероятность <0.6, считаем "неуверенно"
                if emo_prob < 0.6:
                    return "неуверенно", emo_prob
                return emo_ru, emo_prob
            except Exception as e:
                print(f"Ошибка DeepFace: {e}. Fallback на ONNX если доступен.")
                if not hasattr(self, 'session'):
                    return "ошибка", 0.0

        # Fallback на ONNX (если DeepFace не сработал или флаг False)
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))
            gray = gray.astype("float32") / 255.0
            gray = gray[np.newaxis, np.newaxis, :, :]
            outputs = self.session.run(None, {self.input_name: gray})
            logits = outputs[0][0]
            probs = self.softmax(logits)
            idx = int(np.argmax(probs))
            emo_ru = self.emotions_ru[idx]
            emo_prob = float(probs[idx])
            if emo_prob < 0.6:
                return "неуверенно", emo_prob
            return emo_ru, emo_prob
        except Exception as e:
            print(f"Ошибка ONNX: {e}")
            return "ошибка", 0.0