import os
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from process_media import process_image, process_video

TOKEN = ""

DOWNLOAD_DIR = "bot_data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ---------- КНОПКИ ----------
MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("📷 Обработать изображение")],
        [KeyboardButton("🎥 Обработать видео")],
        [KeyboardButton("ℹ️ О проекте")],
        [KeyboardButton("❌ Отмена")]
    ],
    resize_keyboard=True
)

# ---------- СОСТОЯНИЕ ----------
USER_STATE = {}

# ---------- /start ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    USER_STATE[update.effective_user.id] = None
    await update.message.reply_text(
        "👋 Привет!\n\n"
        "Я бот для распознавания действий, эмоций и контекста человека.\n"
        "Выбери, что хочешь сделать:",
        reply_markup=MAIN_KEYBOARD
    )

# ---------- /help ----------
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 Помощь\n\n"
        "/start — главное меню\n"
        "/help — помощь\n\n"
        "Как пользоваться:\n"
        "1️⃣ Выбери тип обработки\n"
        "2️⃣ Отправь фото или видео\n"
        "3️⃣ Получи результат\n\n"
        "❌ Отмена — доступна всегда"
    )

# ---------- ТЕКСТ ----------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    uid = update.effective_user.id

    if text == "📷 Обработать изображение":
        USER_STATE[uid] = "image"
        await update.message.reply_text("📷 Отправь изображение")

    elif text == "🎥 Обработать видео":
        USER_STATE[uid] = "video"
        await update.message.reply_text("🎥 Отправь видео")

    elif text == "ℹ️ О проекте":
        await update.message.reply_text(
            "🔬 Проект:\n"
            "Реальное распознавание действий и намерений человека\n"
            "на основе позы, объектов и эмоций.\n\n"
            "⚙️ Работает на CPU\n"
            "🎯 Для стендов и конференций"
        )

    elif text == "❌ Отмена":
        USER_STATE[uid] = None
        await update.message.reply_text(
            "❌ Действие отменено",
            reply_markup=MAIN_KEYBOARD
        )

    else:
        await update.message.reply_text("❓ Используй кнопки или /help")

# ---------- ФОТО ----------
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if USER_STATE.get(uid) != "image":
        await update.message.reply_text("⚠️ Сначала выбери «Обработать изображение»")
        return

    photo = update.message.photo[-1]
    file = await photo.get_file()

    input_path = os.path.join(DOWNLOAD_DIR, f"{uid}_input.jpg")
    output_path = os.path.join(DOWNLOAD_DIR, f"{uid}_output.jpg")

    await file.download_to_drive(input_path)

    await update.message.reply_text("⏳ Обрабатываю изображение...")
    process_image(input_path, output_path)

    await update.message.reply_photo(
        photo=open(output_path, "rb"),
        caption="✅ Готово!",
        reply_markup=MAIN_KEYBOARD
    )

    USER_STATE[uid] = None

# ---------- ВИДЕО ----------
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if USER_STATE.get(uid) != "video":
        await update.message.reply_text("⚠️ Сначала выбери «Обработать видео»")
        return

    video = update.message.video
    file = await video.get_file()

    input_path = os.path.join(DOWNLOAD_DIR, f"{uid}_input.mp4")
    output_path = os.path.join(DOWNLOAD_DIR, f"{uid}_output.mp4")

    await file.download_to_drive(input_path)

    await update.message.reply_text("⏳ Обрабатываю видео...")
    process_video(input_path, output_path)

    await update.message.reply_video(
        video=open(output_path, "rb"),
        caption="✅ Готово!",
        reply_markup=MAIN_KEYBOARD
    )

    USER_STATE[uid] = None

# ---------- MAIN ----------
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))

    print("🤖 Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
