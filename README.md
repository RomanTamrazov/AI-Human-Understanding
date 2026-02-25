# 🎥 Real-Time Human Action, Emotion & Intent Recognition

Проект по **распознаванию действий, эмоций и намерений человека в реальном времени** с использованием компьютерного зрения и позных моделей.  
Система работает **на CPU**, подходит для демонстрации на стендах, конференциях и легко интегрируется в **Telegram-бота**.

![Demo](assets/demo.gif)


---

## 🚀 Возможности проекта

В реальном времени система умеет:

### 🧍‍♂️ Действия человека
- стоит  
- идёт  
- бежит  
- останавливается  
- наклоняется  
- прыгает / подпрыгивает  
- активно двигается  

### ✋ Жесты (упрощённые, честные)
- поднимает руки  
- машет рукой  
- хлопает  

### 🧠 Намерения (короткий прогноз)
- собирается остановиться  
- продолжит движение  
- собирается прыгнуть  

### 🙂 Эмоции лица
- нейтральная  
- счастлив  
- грустный  
- злой  
- удивлён  
- испуган  
- отвращение  
- презрение  

---

## 🧠 Архитектура системы

Проект построен по **модульному принципу**:

camera / image / video

│

▼

PersonDetector (YOLOv8)

│

▼

PoseEstimator (MediaPipe Pose)

│

├── EmotionRecognizer (ONNX FER+)

│

└── IntentPredictor (кинематика + логика)

│

▼

UI / Video / Telegram Bot


---

## 📦 Используемые технологии

- **Python 3.9+**
- **OpenCV**
- **YOLOv8 (Ultralytics)** — детекция человека
- **MediaPipe Pose & Face Detection**
- **ONNX Runtime** — распознавание эмоций
- **NumPy**
- **Telegram Bot API**

---

## 📥 Установка

### 1️⃣ Клонировать репозиторий
```bash
git clone https://github.com/your_username/Model_real_time_detector.git
cd Model_real_time_detector

python -m venv venv

source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt

python app/main.py

```
---
### 🤖 Telegram-бот
## Бот принимает:
- 📷 изображения
- 🎥 видео
## И возвращает видео/картинку с:
- bounding box человека
- действием
- намерением
- эмоцией
## Запуск:
```bash
python app/bot.py
```
### 👤 Автор
## Роман Тамразов
## ML / Computer Vision
Проект разработан для исследовательских и образовательных целей.
