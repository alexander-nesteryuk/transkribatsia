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
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "4"))
LOCAL_API_URL = os.environ.get("TELEGRAM_LOCAL_API_URL", "")
# Pause between segments (seconds) that triggers a new paragraph
PARAGRAPH_PAUSE_SEC = float(os.environ.get("PARAGRAPH_PAUSE_SEC", "2.0"))

logger.info("Loading Whisper model '%s'...", WHISPER_MODEL)
model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
logger.info("Model loaded.")

semaphore = asyncio.Semaphore(MAX_CONCURRENT)


def transcribe_file(path: str) -> str:
    segments, _ = model.transcribe(
        path,
        beam_size=5,
        vad_filter=True,          # strips silence, improves segmentation
        vad_parameters={"min_silence_duration_ms": 500},
    )

    paragraphs: list[list[str]] = [[]]
    last_end = 0.0

    for segment in segments:
        # Long pause between segments → start a new paragraph
        if paragraphs[-1] and (segment.start - last_end) >= PARAGRAPH_PAUSE_SEC:
            paragraphs.append([])
        paragraphs[-1].append(segment.text.strip())
        last_end = segment.end

    return "\n\n".join(
        " ".join(sentences)
        for sentences in paragraphs
        if sentences
    )


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
        await message.reply_text("Пожалуйста, отправьте аудио или голосое сообщение.")
        return

    status = await message.reply_text("⏳ Обрабатываю...")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        await tg_file.download_to_drive(tmp_path)

        async with semaphore:
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
    builder = Application.builder().token(TELEGRAM_TOKEN)

    if LOCAL_API_URL:
        logger.info("Using local Bot API: %s", LOCAL_API_URL)
        builder = builder.base_url(LOCAL_API_URL).local_mode(True)

    app = builder.build()

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
