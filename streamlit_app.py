import os
import streamlit as st
from pydub import AudioSegment
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs
from streamlit_mic_recorder import mic_recorder

# Load API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

st.title("🩺 AI Doctor with Vision and Voice")

# System Prompt
system_prompt = """You have to act as a professional doctor. I know you are not, but this is for learning purposes. 
With what I see, I think you have .... 
Don't respond as an AI model. Your answer should mimic that of an actual doctor, not an AI bot. 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please."""

# Voice Recording Section
st.write("🎤 Record your voice message:")
audio_data = mic_recorder(start_prompt="Click to Record", key="record")

# ✅ **Show the recorded voice as soon as recording stops**
if audio_data:
    if isinstance(audio_data, dict) and "bytes" in audio_data:
        audio_bytes = audio_data["bytes"]

        # Save recorded audio as WAV
        audio_path = "recorded_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # 🎧 **Show Recorded Voice in UI immediately**
        st.audio(audio_path, format="audio/wav")

    else:
        st.error("Invalid audio data format.")
        st.stop()

# Image Upload Section
image_file = st.file_uploader(
    "📷 Upload a medical image (optional)", type=["png", "jpg", "jpeg"])

if st.button("🩺 Diagnose"):
    if not audio_data:
        st.warning("⚠️ Please record your voice first.")
        st.stop()

    # Transcribe audio
    st.info("📝 Transcribing speech...")
    transcribed_text = transcribe_with_groq(
        GROQ_API_KEY=GROQ_API_KEY, audio_filepath=audio_path, stt_model="whisper-large-v3")
    st.text_area("🗣 Speech to Text Output", transcribed_text)

    # Analyze Image (if provided)
    if image_file:
        image_path = f"temp_image.{image_file.name.split('.')[-1]}"
        with open(image_path, "wb") as f:
            f.write(image_file.read())

        encoded_image = encode_image(image_path)
        doctor_response = analyze_image_with_query(
            query=system_prompt + transcribed_text,
            encoded_image=encoded_image,
            model="llama-3.2-11b-vision-preview"
        )
    else:
        doctor_response = "No image provided for analysis."

    st.text_area("🧑‍⚕️ Doctor's Response", doctor_response)

    # Convert response to speech
    st.info("🔊 Generating voice response...")
    audio_output_path = "doctor_response.mp3"
    text_to_speech_with_elevenlabs(doctor_response, audio_output_path)
    st.audio(audio_output_path, format="audio/mp3")
