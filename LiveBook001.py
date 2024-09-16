import asyncio
from concurrent.futures import ThreadPoolExecutor
from telegram.ext import MessageHandler
from telegram import Audio
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from BotToken import BOT_TOKEN  # Import the bot token from BotToken.py
# Import the OpenAI API key from OpenAIAPI.py
from OpenAIAPI import OPENAI_API_KEY
import logging
import requests
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ConversationHandler, MessageHandler, filters, CallbackContext

import openai
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from striprtf.striprtf import rtf_to_text  # Import rtf_to_text from striprtf

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Initialize the bot with the imported token
bot = Bot(token=BOT_TOKEN)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Conversation states for the bot
QUERY = range(1)

# Load the RTF file using striprtf


def load_rtf_file(file_path):
    with open(file_path, 'r') as file:
        rtf_content = file.read()
    return rtf_to_text(rtf_content)

# Tokenize and embed the content


def embed_text(text, model="text-embedding-3-large"):
    # Pass the API key directly to the client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    # Access the embedding correctly
    return response.data[0].embedding


# Load and preprocess the dataset

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the dataset file directly in the root
dataset_path = os.path.join(current_dir, 'Dataset.txt')
print(f"\nDataset Path: {dataset_path}")
# Load and preprocess the dataset
full_text = load_rtf_file(dataset_path)

# Split the text into chunks


def split_text(text, max_tokens=2000, overlap_tokens=300):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    start_index = 0

    while start_index < len(tokens):
        # Define the end index of the current chunk
        end_index = start_index + max_tokens

        # Slice the tokens to create a chunk
        chunk = tokens[start_index:end_index]
        chunks.append(tokenizer.decode(chunk))

        # Move the start index forward by max_tokens - overlap_tokens to create overlap
        start_index += max_tokens - overlap_tokens

    return chunks

# Load and preprocess the dataset


# Load embeddings if they exist, otherwise embed the text
embeddings_file = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "embeddings.npy")
if os.path.exists(embeddings_file):
    embeddings = np.load(embeddings_file)
else:
    text_chunks = split_text(full_text)
    embeddings = [embed_text(chunk) for chunk in text_chunks]
    np.save(embeddings_file, embeddings)

# Global variable for text chunks
text_chunks = split_text(full_text)


# Retrieve the top 5 most relevant chunks
def retrieve_relevant_data(query, top_k=5):
    global text_chunks
    query_embedding = embed_text(query)
    similarities = cosine_similarity([query_embedding], embeddings)
    # Get the indices of the top k most similar chunks
    top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
    # Combine the top k chunks
    relevant_chunks = "\n\n".join([text_chunks[i] for i in top_k_indices])
    return relevant_chunks


# Generate a response using OpenAI

def generate_response(query, context_chunk):
    prompt = f"The Retrieved Chunks (based on the user's query):\n{context_chunk}\n\nBased on the retrieved chunks above, answer the user's query from the Molecular Cell Biology textbook and your own valid knowledge by Baltimore and Lodish, please.\nAbout BioMindBot (you): BioMindBot is your AI-powered guide for mastering Molecular Cell Biology, offering insights and answers based on the renowned textbook by Baltimore & Lodish. Whether you're a student, educator, or biology enthusiast, BioMindBot is here to help you explore and understand complex cellular mechanisms and molecular interactions.\n\nQuery: {query}\n\nResponse:"
    print(f"\n\n\nPrompt:\n{prompt}")
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }
    data = {
        'model': 'gpt-4o-mini',  # or any other model you'd like to use
        'messages': [
            {"role": "system", "content": "You are a helpful assistant and the master of the Molecular Cell Biology textbook by Baltimore and Lodish."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(
        'https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        logging.error(
            f"OpenAI API request failed with status code {response.status_code}: {response.text}")
        return None


# Function to convert MP3 to OGG

def convert_mp3_to_ogg(mp3_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_ogg_file:
        temp_ogg_path = temp_ogg_file.name

    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(temp_ogg_path, format="ogg", codec="libopus")

    return temp_ogg_path

# Updated text_to_speech function


def text_to_speech(text):
    # Create a temporary MP3 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_path = temp_audio_file.name

    # Generate the speech and save it as MP3
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text
    ) as response:
        response.stream_to_file(temp_audio_path)

    # Convert the MP3 file to OGG
    temp_ogg_path = convert_mp3_to_ogg(temp_audio_path)

    # Remove the temporary MP3 file after conversion
    os.remove(temp_audio_path)

    return temp_ogg_path

# Start command


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Welcome to BioMindBot, your guide to Molecular Cell Biology. Please enter your query or request.")
    return QUERY

# Handle text queries


# Function to convert any audio format to WAV
def convert_audio_to_wav(audio_file_path):
    wav_file_path = f"{os.path.splitext(audio_file_path)[0]}.wav"

    # Check if the file exists and has a valid size
    if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
        raise RuntimeError(
            f"File {audio_file_path} does not exist or is empty.")

    try:
        # Attempt to load the OGG file using pydub
        audio = AudioSegment.from_file(audio_file_path, format="ogg")
        audio.export(wav_file_path, format="wav")
        return wav_file_path
    except Exception as e:
        raise RuntimeError(f"Error converting file {audio_file_path}: {e}")


# Function to transcribe voice to text


def transcribe_voice_to_text(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(
                audio_data, language="en-US")  # You can modify the language as needed
            return transcription
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand your voice."
        except sr.RequestError as e:
            return f"Error connecting to the voice service: {e}"

# Updated query_handler to support voice queries


# Initialize an executor for running blocking tasks
executor = ThreadPoolExecutor()


async def query_handler(update: Update, context: CallbackContext):
    # Check if it's a voice message
    if update.message.voice:
        await update.message.reply_text("Processing your voice query, please wait...")

        # Download the voice file
        voice_file = await update.message.voice.get_file()

        # Save the voice file temporarily
        temp_ogg_path = os.path.join(os.getcwd(), "voice_input.ogg")
        await voice_file.download_to_drive(temp_ogg_path)

        try:
            # Convert voice file to WAV (blocking operation, so run it in the executor)
            wav_file_path = await asyncio.get_running_loop().run_in_executor(executor, convert_audio_to_wav, temp_ogg_path)

            # Transcribe the voice to text (blocking operation, so run it in the executor)
            user_query = await asyncio.get_running_loop().run_in_executor(executor, transcribe_voice_to_text, wav_file_path)

            # Inform the user about the transcription result
            if not user_query:
                await update.message.reply_text("Sorry, I couldn't understand your voice. Please try again.")
                return
            else:
                await update.message.reply_text(f"Transcribed voice query: {user_query}")

            # Clean up the temporary files
            os.remove(temp_ogg_path)
            os.remove(wav_file_path)

        except Exception as e:
            await update.message.reply_text(f"Error processing voice file: {e}")
            return
    else:
        # If it's a text message, just get the text directly
        user_query = update.message.text
        await update.message.reply_text("Your text query is being processed, please wait...")

    # Run the data retrieval and response generation in a separate thread to avoid blocking
    loop = asyncio.get_running_loop()

    # Retrieve the top 3 most relevant chunks of data
    retrieved_chunks = await loop.run_in_executor(executor, retrieve_relevant_data, user_query)

    # Generate a response based on the retrieved data
    response_text = await loop.run_in_executor(executor, generate_response, user_query, retrieved_chunks)

    # Send the text response back to the user
    if response_text:
        await update.message.reply_text(f"Response:\n\n{response_text}")

        # Convert the response text to speech (in a separate thread since it may block)
        speech_file_path = await loop.run_in_executor(executor, text_to_speech, response_text)

        # Send the MP3 file to the user asynchronously
        with open(speech_file_path, 'rb') as audio:
            await update.message.reply_audio(audio)

        # Delete the temporary file after sending
        os.remove(speech_file_path)
    else:
        await update.message.reply_text("Failed to generate a response. Please try again.")

    return QUERY  # Keep the conversation going


# Cancel command


async def cancel(update: Update, context: CallbackContext):
    await update.message.reply_text('Operation canceled. You can start again by typing /start.')
    return ConversationHandler.END


def main():
    application = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            QUERY: [MessageHandler(filters.TEXT & ~filters.COMMAND, query_handler),
                    MessageHandler(filters.VOICE, query_handler)],  # Handle voice inputs
        },
        # Now the cancel handler is defined
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    application.add_handler(conv_handler)
    application.run_polling()


if __name__ == "__main__":
    main()
