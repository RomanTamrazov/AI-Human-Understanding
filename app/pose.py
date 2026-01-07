import mediapipe as mp
import cv2

class PoseEstimator:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False
        )
        self.face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # Для дальних лиц
            min_detection_confidence=0.4  # Пониженный порог для лучшей детекции
        )

    def estimate(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks.landmark

    def detect_face(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face.process(rgb)
        if not result.detections:
            return None
        det = result.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1 = int(max(0, bbox.xmin) * w)
        y1 = int(max(0, bbox.ymin) * h)
        x2 = int(min(1, bbox.xmin + bbox.width) * w)
        y2 = int(min(1, bbox.ymin + bbox.height) * h)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2