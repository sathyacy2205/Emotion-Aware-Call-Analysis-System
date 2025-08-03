import streamlit as st
import os

def read_api_key():
    try:
        with open("apikey.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.write("Error: apikey.txt not found in the current directory.")
        exit(1)

def Home():
    markdown_file_path = "readme.md"
    with open(markdown_file_path, "r", encoding="utf-8") as file:
        md_content = file.read()
    st.markdown(md_content, unsafe_allow_html=True)
    return

def Speech_Recognizer():
    import speech_recognition as sr
    from pydub import AudioSegment

    def prepare_voice_file(path: str) -> str:
        if os.path.splitext(path)[1] == '.wav':
            return path
        elif os.path.splitext(path)[1] in ('.mp3', '.m4a', '.ogg', '.flac'):
            audio_file = AudioSegment.from_file(path, format=os.path.splitext(path)[1][1:])
            wav_file = os.path.splitext(path)[0] + '.wav'
            audio_file.export(wav_file, format='wav')
            return wav_file
        else:
            raise ValueError(f'Unsupported audio format: {path}')

    def transcribe_audio(audio_data, language) -> str:
        r = sr.Recognizer()
        return r.recognize_google(audio_data, language=language)

    def write_transcription_to_file(text, output_file) -> None:
        with open(output_file, 'w') as f:
            f.write(text)

    def speech_to_text(input_path: str, output_path: str, language: str) -> None:
        wav_file = prepare_voice_file(input_path)
        with sr.AudioFile(wav_file) as source:
            audio_data = sr.Recognizer().record(source)
            text = transcribe_audio(audio_data, language)
            write_transcription_to_file(text, output_path)
            st.write('Transcription:')
            st.write(text)

    uploaded_files = st.file_uploader("Choose audio file", type=["wav", "mp3", "flac", "ogg"], accept_multiple_files=True)
    os.makedirs("uploads", exist_ok=True)

    for uploaded_file in uploaded_files:
        st.write("filename:", uploaded_file.name)
        input_path = os.path.join("uploads", uploaded_file.name)

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(input_path, format="audio/mpeg", loop=False)

        if not os.path.isfile(input_path):
            st.markdown('**Error:** File not found.')
        else:
            output_path = "output.txt"
            language = "en-US"
            try:
                speech_to_text(input_path, output_path, language)
            except Exception as e:
                st.markdown(f'**Error:** {e}')

def Emotion_Analyzer():
    from openai import OpenAI
    key = open("aimlapi_key.txt", "r").read()
    base_url = "https://api.aimlapi.com/v1"
    system_prompt = "identify emotion and severity of the sentence on a scale of 1-10 along with an adjective (anger,fear,panic,neutral, etc):"
    st.title("Emotion Analysis (API)")
    user_prompt = open("output.txt", "r").read()
    prompt = system_prompt + user_prompt
    st.markdown("**User:** " + user_prompt)

    try:
        api = OpenAI(api_key=key, base_url=base_url)
        completion = api.chat.completions.create(
            model="google/gemma-3n-e4b-it",
            messages=[{"role": "user", "content": prompt}]
        )
        response = completion.choices[0].message.content
        st.write("**RESULT**:")
        st.markdown(response)
    except Exception as e:
        st.markdown(f"**error**: {e}")

def TTS():
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)

    text = st.text_input("Text", "officer, I drop kicked that child on act of self defense")
    audiofile = st.text_input("File Name", "audio_x")

    engine.save_to_file(text, audiofile + ".wav")
    engine.runAndWait()

    with open(audiofile + ".wav", "rb") as file:
        audio_bytes = file.read()

    st.download_button("📥 Download WAV Audio", data=audio_bytes, file_name=audiofile + ".wav", mime="audio/wav")
    st.audio(audio_bytes, format="audio/wav")

def Local_Machine_Learning():
    import joblib

    vectorizer = joblib.load('tfidf_vectorizer_fin.pkl')
    model = joblib.load('emotion_model_xgb.pkl')

    new_text = open("output.txt", "r").read()
    st.markdown("## User: \n  ###\t " + new_text + "\n\n")
    new_vec = vectorizer.transform([new_text])

    label_map = {
        0: 'anger', 1: 'anxiety', 2: 'calm', 3: 'confusion', 4: 'desperation',
        5: 'fear', 6: 'frustration', 7: 'joy', 8: 'neutral',
        9: 'panic', 10: 'sadness', 11: 'shock', 12: 'tension'
    }

    predicted_label = model.predict(new_vec)[0]
    st.markdown(f"""
    <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;">
        <h3 style="color:#2c3e50;">  Emotion Prediction Result</h3>
        <p><strong>🔢 Predicted Label:</strong> <code>{predicted_label}</code></p>
        <p><strong>🎭 Predicted Emotion:</strong> <span style="color:#007ACC;font-weight:bold;">{label_map[predicted_label].capitalize()}</span></p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Emotion Label Mapping")
    st.json(label_map)

def Language_Detection():
    import streamlit as st
    import whisper
    import os
    from pydub import AudioSegment

    def prepare_voice_file(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.wav':
            return path
        elif ext in ('.mp3', '.m4a', '.ogg', '.flac'):
            audio = AudioSegment.from_file(path, format=ext[1:])
            wav_path = os.path.splitext(path)[0] + '.wav'
            audio.export(wav_path, format='wav')
            return wav_path
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def detect_language_only(input_path: str):
        model = whisper.load_model("base")
        audio = whisper.load_audio(input_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        st.success(f"🗣️ Detected Language: `{detected_lang}`")

    st.title("🌍 Language Detection (Only)")

    uploaded_files = st.file_uploader("📤 Upload audio file", type=["mp3", "wav", "m4a", "flac", "ogg"], accept_multiple_files=True)
    os.makedirs("uploads", exist_ok=True)

    for uploaded_file in uploaded_files:
        st.write(f"**File Uploaded:** `{uploaded_file.name}`")
        st.audio(uploaded_file, format="audio/mpeg", loop=False)

        input_path = os.path.join("uploads", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            wav_file = prepare_voice_file(input_path)
            detect_language_only(wav_file)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def Transcription_Translation():
    import whisper
    from pydub import AudioSegment

    def prepare_voice_file(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.wav':
            return path
        elif ext in ('.mp3', '.m4a', '.ogg', '.flac'):
            audio = AudioSegment.from_file(path, format=ext[1:])
            wav_path = os.path.splitext(path)[0] + '.wav'
            audio.export(wav_path, format='wav')
            return wav_path
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def transcribe_and_translate(input_path: str):
        model = whisper.load_model("medium")
        audio = whisper.load_audio(input_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        st.markdown(f"🔍 **Detected Language:** `{detected_lang}`")

        transcription = model.transcribe(input_path, language=detected_lang, fp16=False)
        st.markdown("### 📝 Transcription in Native Language:")
        st.write(transcription["text"])

        if detected_lang in ["ta", "te"]:
            translation = model.transcribe(input_path, task="translate", language=detected_lang, fp16=False)
            st.markdown("### 🌐 Translated to English:")
            st.write(translation["text"])
            with open("output.txt", 'w', encoding='utf-8') as f:
                f.write(translation["text"])
            st.success("✔️ Translated text saved to `output.txt`.")
        else:
            st.info("🌐 Skipped translation: Not Tamil or Telugu.")
            with open("output.txt", 'w', encoding='utf-8') as f:
                f.write(transcription["text"])

    st.title("🎙️ Language Detection & Translation")
    uploaded_files = st.file_uploader("📤 Upload audio file", type=["mp3", "wav", "m4a", "flac", "ogg"], accept_multiple_files=True)
    os.makedirs("uploads", exist_ok=True)

    for uploaded_file in uploaded_files:
        input_path = os.path.join("uploads", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"**File Uploaded:** `{uploaded_file.name}`")
        st.audio(input_path, format="audio/mpeg", loop=False)

        try:
            wav_path = prepare_voice_file(input_path)
            transcribe_and_translate(wav_path)
            st.markdown("---")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Sidebar Navigation
page_names_to_funcs = {
    "Home": Home,
    "Transcription Translation": Transcription_Translation,
    "Language Detection": Language_Detection,
    "Local ML": Local_Machine_Learning,
    "Text to Speech": TTS,
    # "Speech Recognition": Speech_Recognizer,
    "Emotion Analyzer": Emotion_Analyzer,
}
st.sidebar.markdown("# EMOLYZER")
st.sidebar.image("./emo.png")
demo_name = st.sidebar.selectbox("What you want to do?", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
