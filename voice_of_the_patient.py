# if you don't use pipenv, uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

import os
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from groq import Groq

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Records audio from the microphone and saves it as an MP3 file.

    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_limit (int): Maximum duration of the recorded phrase (in seconds).
    """
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")

            # Record the audio
            audio_data = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")

            # Convert the recorded audio to an MP3 file
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")

            logging.info(f"Audio saved to {file_path}")

    except Exception as e:
        logging.error(f"An error occurred while recording: {e}")


def transcribe_with_groq(GROQ_API_KEY, audio_filepath, stt_model="whisper-large-v3"):
    """
    Transcribes an audio file using Groq's Whisper model.

    Args:
    GROQ_API_KEY (str): API key for authentication.
    audio_filepath (str): Path to the audio file.
    stt_model (str): Whisper model name (default: whisper-large-v3).

    Returns:
    str: Transcribed text.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)

        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )

        return transcription.text

    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}")
        return None


# if __name__ == "__main__":
#     # Test recording and transcription
#     GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
#     audio_filepath = "patient_voice_test_for_patient.mp3"

#     record_audio(file_path=audio_filepath)

#     if os.path.exists(audio_filepath):
#         transcription = transcribe_with_groq(GROQ_API_KEY, audio_filepath)
#         logging.info(f"Transcription: {transcription}")
