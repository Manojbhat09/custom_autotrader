from telegram import Bot
import asyncio
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging 
# Enable logging
# Create a logger object
logger = logging.getLogger('telegram')
logger.setLevel(logging.INFO)  # Set the logging level

# Create a file handler to write logs to a file
file_handler = logging.FileHandler(f'telegram.log')
file_formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Create a console handler to print logs to the console
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
logger.propagate = False 

# Initialize the bot with your token
token = '6766648286:AAFHVOaJoZxX0RDYeOBpvGsoN2pz9iFKDUQ'
bot = Bot(token)

# Function to send a message
# Asynchronous function to send a message
async def send_telegram_message(message, chat_id= '5339088967'):
    await bot.send_message(chat_id=chat_id, text=message)

# Function to send a message aand start polling for the first response
# Asynchronous function to send a message
async def send_telegram_message(message, chat_id= '5339088967'):
    await bot.send_message(chat_id=chat_id, text=message)
    start_polling()

def start_polling():
    """Start the bot."""
    updater = Updater(token, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Handle incoming messages
    dp.add_handler(MessageHandler(Filters.text, handle_message))

    # Log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT
    updater.idle()

# Function to handle errors
def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning(f'Update "{update}" caused error "{context.error}"')

# Function to handle messages
def handle_message(update, context):
    """Echo the user message."""
    incoming_message = update.message.text
    chat_id = update.message.chat_id
    logger.info(f"Received message: {incoming_message}")
    # Here you can add your logic to re
    return incoming_message

if __name__ == "__main__":
    # Example usage
    chat_id = '5339088967'  # Replace with the chat ID where you want to send the message
    message = 'Hello, this is a test message.'
    asyncio.run(send_telegram_message(chat_id, message))

    # curl https://api.telegram.org/bot6766648286:AAFHVOaJoZxX0RDYeOBpvGsoN2pz9iFKDUQ/getUpdates