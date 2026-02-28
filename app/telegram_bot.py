"""
telegram_bot.py – Async Telegram Bot for scAlerite
===================================================

🧠 HOW THE BOT WORKS:
─────────────────────
1. User sends a message on Telegram.
2. python-telegram-bot receives it asynchronously.
3. Bot forwards the question to our FastAPI backend via HTTP POST.
4. FastAPI searches the FAISS vector store and returns matching chunks.
5. Bot formats the response and sends it back to the user.

This is the LAST piece of the RAG pipeline!
"""

import os
import logging
import httpx
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ─────────────────────────────────
# Suggested Questions (Quick Buttons)
# ─────────────────────────────────
SUGGESTED_QUESTIONS = [
    ["📅 When does the semester start?", "📜 What is the attendance policy?"],
    ["🏢 How to book a meeting room?", "💰 What is the fee structure?"],
    ["🏭 What is the industry immersion policy?", "📖 What is the curriculum?"],
    ["🎓 What is the placement policy?", "🏠 What are the hostel rules?"],
]

def get_suggested_keyboard():
    """Build the reply keyboard with suggested questions."""
    return ReplyKeyboardMarkup(
        SUGGESTED_QUESTIONS,
        resize_keyboard=True,       # Shrink buttons to fit
        one_time_keyboard=False,    # Keep keyboard visible
    )

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────
# Configuration
# ─────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────
# Command Handlers
# ─────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command — shows welcome + suggested question buttons."""
    welcome_message = (
        "👋 Welcome to *scAlerite* — your college academic assistant!\n\n"
        "Tap any question below or type your own! 👇"
    )
    await update.message.reply_text(
        welcome_message,
        parse_mode="Markdown",
        reply_markup=get_suggested_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = (
        "🆘 *How to use scAlerite:*\n\n"
        "Simply type any academic question, for example:\n"
        '• "When does the semester start?"\n'
        '• "What is the attendance policy?"\n'
        '• "How to book a meeting room?"\n\n'
        "I'll search through all college policy documents and return "
        "the most relevant information.\n\n"
        "*Commands:*\n"
        "/start — Welcome message\n"
        "/help  — This help text"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


# ─────────────────────────────────
# Message Handler (Core Logic)
# ─────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle any text message from the user.
    Sends the question to FastAPI and returns the answer.
    """
    user_question = update.message.text
    user_name = update.effective_user.first_name
    logger.info(f"Question from {user_name}: {user_question}")

    # Show "typing..." indicator while processing
    await update.message.chat.send_action("typing")

    try:
        # Send question to FastAPI backend
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/query",
                json={"question": user_question, "top_k": 3},
            )

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer found.")

            # Telegram has a 4096 char limit per message
            if len(answer) > 4000:
                answer = answer[:4000] + "\n\n... (truncated)"

            await update.message.reply_text(
                f"🔍 *Results for:* _{user_question}_\n\n{answer}",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                "⚠️ Sorry, the backend returned an error. Please try again later."
            )
            logger.error(f"Backend error: {response.status_code} - {response.text}")

    except httpx.ConnectError:
        await update.message.reply_text(
            "🔌 The backend server is not running. "
            "Please make sure to start it with:\n"
            "`python3 app/main.py`",
            parse_mode="Markdown",
        )
        logger.error("Could not connect to FastAPI backend.")

    except Exception as e:
        await update.message.reply_text(
            "❌ An unexpected error occurred. Please try again."
        )
        logger.error(f"Unexpected error: {e}", exc_info=True)


# ─────────────────────────────────
# Error Handler
# ─────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log errors caused by updates."""
    logger.error(f"Update {update} caused error: {context.error}")


# ─────────────────────────────────
# Main Entry Point
# ─────────────────────────────────

def main():
    """Start the Telegram bot."""
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_TOKEN not found!")
        print("   Create a .env file with: TELEGRAM_TOKEN=your_token_here")
        return

    print("🤖 Starting scAlerite Telegram Bot...")
    print(f"🔗 Backend URL: {BACKEND_URL}")

    # Build the application
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Register error handler
    application.add_error_handler(error_handler)

    # Start polling for messages
    print("✅ Bot is running! Send a message on Telegram.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
