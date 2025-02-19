import os
import logging
import sounddevice as sd
import numpy as np
import wave
from groq import Groq
from pydub import AudioSegment
from dotenv import load_dotenv

# Load environment variables (if not using pipenv)
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def record_audio(file_path, duration=5, samplerate=44100):
    """
    Records audio using sounddevice and saves it as an MP3 file.

    Args:
    file_path (str): Path to save the recorded audio.
    duration (int): Recording duration in seconds.
    samplerate (int): Sample rate (default: 44100 Hz).
    """
    try:
        logging.info("Recording started... Speak now!")
        audio_data = sd.rec(int(duration * samplerate),
                            samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()  # Wait until recording is finished
        logging.info("Recording finished.")

        # Save as a WAV file
        wav_file = file_path.replace(".mp3", ".wav")
        with wave.open(wav_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())

        # Convert WAV to MP3
        audio_segment = AudioSegment.from_wav(wav_file)
        audio_segment.export(file_path, format="mp3", bitrate="128k")
        os.remove(wav_file)  # Remove temporary WAV file
        logging.info(f"Audio saved to {file_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


# Step 2: Setup Speech-to-Text (STT) Model
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
stt_model = "whisper-large-v3"


def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
    """
    Transcribes an audio file using Groq API.

    Args:
    stt_model (str): Model to use for transcription.
    audio_filepath (str): Path to the audio file.
    GROQ_API_KEY (str): API key for authentication.

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
        logging.error(f"Error in transcription: {e}")
        return ""


# Example Usage
# audio_filepath = "patient_voice_test.mp3"
# record_audio(audio_filepath, duration=10)  # Record for 10 seconds
# transcribed_text = transcribe_with_groq(
#     stt_model, audio_filepath, GROQ_API_KEY)
# print("Transcription:", transcribed_text)
