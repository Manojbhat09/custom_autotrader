from telegram import Bot
import asyncio
from telegram.ext import  Updater, CommandHandler, MessageHandler, filters, Filters
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
update_queue = asyncio.Queue()

# Global variable to store message
CHAT_ID = '5339088967'  # Replace with the chat ID where you want to send the message
incoming_messages = []
is_polling = False

# Function to process incoming messages
def handle_message(bot, update):
    global incoming_messages
    incoming_messages.append(update)

# Function to start polling
def start_polling():
    global is_polling
    if not is_polling:
        logger.info("polling")
        updater = Updater(token)
        dp = updater.dispatcher
        dp.add_handler(MessageHandler(Filters.text, handle_message))
        updater.start_polling()
        is_polling = True
        return updater
    return None


def process_message(message):
    proceed, response = False, ""
    # Conditional logic based on the message
    if message.lower() == 'done' or message.lower() == 'yes':
        # Add your logic here for the "Done" message
        response = "Received 'Done', Great! You made the trade "
    elif message.lower() == 'no':
        # Add your logic here for the "No" message
        response = "Received 'No', Ignoring the alert "
    elif 'you' in message.lower() or message is None:
        response = "anyways you want me to handle it, let me execute the trade thnx"
        proceed = True
    else:
        response = "Message not recognized. Should have responded with Done Yes or No, moving on"
        proceed = True
    return response, proceed

# Asynchronous function to wait for a response
async def wait_for_response(timeout=60):
    start_time = asyncio.get_event_loop().time()
    while True:
        if incoming_messages:
            logger.info("found messages")
            update_object = incoming_messages.pop(0)
            response, proceed = process_message( update_object.message.text )
            bot.send_message(chat_id=CHAT_ID, text=response)
            return proceed
        if (asyncio.get_event_loop().time() - start_time) > timeout:
            return False
        await asyncio.sleep(0.5)  # Sleep briefly to yield control

async def send_telegram_message_poll(message):
    global is_polling
    if not is_polling:
        updater = start_polling()
    bot.send_message(chat_id=CHAT_ID, text=message)
    result = await wait_for_response()
    if is_polling and updater is not None:
        updater.stop()
        is_polling = False
    return result

async def main():
    # Example usage
    chat_id = CHAT_ID # Replace with your actual chat ID
    message = 'Stonks is up, start tradeeeee'
    response = await send_telegram_message_poll(message)
    print(f"Action to take: {response}")


if __name__ == "__main__":
    # Example usage
    message = 'Stonks is up, start tradeeeee'
    # asyncio.run(send_telegram_message(chat_id, message))
    asyncio.run(main())

    # curl https://api.telegram.org/bot6766648286:AAFHVOaJoZxX0RDYeOBpvGsoN2pz9iFKDUQ/getUpdates