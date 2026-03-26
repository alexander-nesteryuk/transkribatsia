import asyncio
import logging
import os
import tempfile
from functools import partial
from pathlib import Path

from faster_whisper import WhisperModel
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

logger.info("Loading Whisper model '%s'...", WHISPER_MODEL)
model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
logger.info("Model loaded.")


def transcribe_file(path: str) -> str:
    segments, _ = model.transcribe(path, beam_size=5)
    return " ".join(segment.text.strip() for segment in segments).strip()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! 🎙️\n\n"
        "Отправь мне голосовое сообщение или аудиофайл — "
        "и я верну текстовую транскрипцию.\n\n"
        "Поддерживаемые форматы: голосовые сообщения, MP3, M4A, WAV, OGG, видеосообщения."
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message

    if message.voice:
        tg_file = await message.voice.get_file()
        suffix = ".ogg"
    elif message.audio:
        tg_file = await message.audio.get_file()
        suffix = Path(message.audio.file_name or "audio.mp3").suffix or ".mp3"
    elif message.video_note:
        tg_file = await message.video_note.get_file()
        suffix = ".mp4"
    elif message.video:
        tg_file = await message.video.get_file()
        suffix = ".mp4"
    else:
        await message.reply_text("Пожалуйста, отправьте аудио или голосовое сообщение.")
        return

    status = await message.reply_text("⏳ Обрабатываю...")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        await tg_file.download_to_drive(tmp_path)

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, partial(transcribe_file, tmp_path))

        if not text:
            await status.edit_text("Не удалось распознать речь. Попробуйте другой файл.")
            return

        await status.edit_text(text)

    except Exception as exc:
        logger.exception("Transcription error: %s", exc)
        await status.edit_text("❌ Ошибка при обработке. Попробуйте ещё раз.")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(
            filters.VOICE | filters.AUDIO | filters.VIDEO_NOTE | filters.VIDEO,
            handle_audio,
        )
    )

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
