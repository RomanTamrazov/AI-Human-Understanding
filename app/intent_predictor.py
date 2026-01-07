import numpy as np
from collections import deque
import math

class IntentPredictor:
    def __init__(self):
        self.positions = deque(maxlen=25)
        self.body_speed_hist = deque(maxlen=10)
        self.hand_hist = deque(maxlen=10)
        self.last_action = "стоит"

    # -------------------- UPDATE --------------------
    def update(self, bbox, pose):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if self.positions:
            px, py = self.positions[-1]
            speed = math.hypot(cx - px, cy - py)
            self.body_speed_hist.append(speed)

        self.positions.append((cx, cy))

        # движение рук
        lw, rw = pose[15], pose[16]
        self.hand_hist.append(abs(lw.y - rw.y))

    # -------------------- FEATURES --------------------
    def body_speed(self):
        if not self.body_speed_hist:
            return 0.0
        return np.mean(self.body_speed_hist)

    def hand_activity(self):
        if len(self.hand_hist) < 2:
            return 0.0
        return np.std(self.hand_hist)

    def vertical_motion(self):
        if len(self.positions) < 2:
            return 0.0
        return self.positions[-2][1] - self.positions[-1][1]

    # -------------------- ACTION --------------------
    def detect_action(self, pose):
        speed = self.body_speed()
        hands = self.hand_activity()
        vert = self.vertical_motion()

        ls, rs = pose[11], pose[12]
        lw, rw = pose[15], pose[16]
        nose = pose[0]

        scores = {}

        # ----------- ПОЗА СТОЯ / ДВИЖЕНИЕ -----------
        if speed < 1.5:
            scores["стоит"] = 0.9

        if 1.5 <= speed < 4:
            scores["идёт"] = 0.7

        # бег ТОЛЬКО если:
        # 1) высокая скорость
        # 2) мало движения рук
        # 3) устойчиво несколько кадров
        if speed >= 4.5 and hands < 0.015:
            scores["бежит"] = min(speed / 8, 0.9)

        # ----------- ЖЕСТЫ РУК -----------
        if hands > 0.03 and speed < 3:
            scores["машет рукой"] = 0.8

        if abs(lw.y - rw.y) < 0.03 and hands > 0.04:
            scores["хлопает"] = 0.9

        if lw.y < ls.y and rw.y < rs.y:
            scores["absolute cinema"] = 0.85

        # ----------- КОРПУС -----------
        if nose.y > (ls.y + rs.y) / 2:
            scores["наклоняется"] = 0.7

        # ----------- ПРЫЖКИ -----------
        if vert > 6 and speed > 3:
            scores["прыгает"] = 0.9
        elif vert > 3:
            scores["подпрыгивает"] = 0.6

        # ----------- АКТИВНОСТЬ -----------
        if speed > 2 or hands > 0.02:
            scores["активно двигается"] = 0.6

        # ----------- ВЫБОР -----------
        if not scores:
            action = self.last_action
            conf = 0.4
        else:
            action = max(scores, key=scores.get)
            conf = scores[action]

        # защита от резких скачков
        if action != self.last_action and conf < 0.65:
            action = self.last_action
            conf *= 0.9

        self.last_action = action
        return action, round(min(conf, 0.95), 2)

    # -------------------- INTENT --------------------
    def detect_intent(self, pose):
        speed = self.body_speed()
        vert = self.vertical_motion()

        if speed < 1.5:
            return "собирается остановиться", 0.8
        if vert > 3:
            return "собирается прыгнуть", 0.75
        if speed > 3:
            return "продолжит движение", 0.85

        return "анализ...", 0.4
