import streamlit as st

def read_api_key():
    """Reads the API key from apikey.txt in the current directory."""
    try:
        with open("apikey.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.write("Error: apikey.txt not found in the current directory.")
        exit(1)

def Home():
    import streamlit as st
    # Path to your markdown file
    markdown_file_path = "readme.md"

    # Read the file contents
    with open(markdown_file_path, "r", encoding="utf-8") as file:
        md_content = file.read()

    # Display the markdown content in the app
    st.markdown(md_content, unsafe_allow_html=True)
    return 

def Speech_Recognizer():
    import os
    import subprocess
    import os
    import speech_recognition as sr
    from pydub import AudioSegment

    def prepare_voice_file(path: str) -> str:
        """
        Converts the input audio file to WAV format if necessary and returns the path to the WAV file.
        """
        if os.path.splitext(path)[1] == '.wav':
            return path
        elif os.path.splitext(path)[1] in ('.mp3', '.m4a', '.ogg', '.flac'):
            audio_file = AudioSegment.from_file(
                path, format=os.path.splitext(path)[1][1:])
            wav_file = os.path.splitext(path)[0] + '.wav'
            audio_file.export(wav_file, format='wav')
            return wav_file
        else:
            raise ValueError(
                f'Unsupported audio format: {format(os.path.splitext(path)[1])}')


    def transcribe_audio(audio_data, language) -> str:
        """
        Transcribes audio data to text using Google's speech recognition API.
        """
        r = sr.Recognizer()
        text = r.recognize_google(audio_data, language=language)
        return text


    def write_transcription_to_file(text, output_file) -> None:
        """
        Writes the transcribed text to the output file.
        """
        with open(output_file, 'w') as f:
            f.write(text)


    def speech_to_text(input_path: str, output_path: str, language: str) -> None:
        """
        Transcribes an audio file at the given path to text and writes the transcribed text to the output file.
        """
        wav_file = prepare_voice_file(input_path)
        with sr.AudioFile(wav_file) as source:
            audio_data = sr.Recognizer().record(source)
            text = transcribe_audio(audio_data, language)
            write_transcription_to_file(text, output_path)
            st.write('Transcription:')
            st.write(text)


    
    uploaded_files = st.file_uploader(
        "Choose a CSV file", accept_multiple_files=True
    )
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        
        st.audio(uploaded_file.name, format="audio/mpeg", loop=False)

        input_path = uploaded_file.name
        with open(os.path.join("",uploaded_file.name),"wb") as f:
           f.write(uploaded_file.getbuffer())
        st.write(input_path)
        if not os.path.isfile(input_path):
            st.markdown('**Error:** File not found.')
            exit(1)
        else:
            output_path = "output.txt"
            language = "en-US"
            try:
                speech_to_text(input_path, output_path, language)
            except Exception as e:
                st.markdown('**Error:** {e}')
                exit(1)


def Emotion_Analyzer():
    from openai import OpenAI

    file=open("aimlapi_key.txt","r")
    key=file.read()
    
    base_url = "https://api.aimlapi.com/v1"
    api_key = key
    system_prompt = "identify emotion and serverity of the sentence on a scale of 1-10 along with an adjective (anger,fear,panic,neutral, etc):"

    st.title("Emotion Analysis: (API)")

    #user_prompt=input("User:")
    user_prompt=open("output.txt","r").read()

    prompt=system_prompt+user_prompt

    st.markdown("**User:** "+user_prompt)

    try:

        api = OpenAI(api_key=api_key, base_url=base_url)

        completion = api.chat.completions.create(
            model="google/gemma-3n-e4b-it",
            messages=[
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        response = completion.choices[0].message.content
        st.write("**RESULT**:")
        st.markdown(response)
    except Exception as e:
        st.markdown("**error**:",e)

def TTS():
    import pyttsx3
    import streamlit as st

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)    # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

    # Choose voice (optional)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Change index if needed

    # Text to speak
    # text = "officer, i drop kicked that child on act of self defence"
    text=st.text_input("Text","officer, i drop kicked that child on act of self defense")

    audiofile=st.text_input("File Name","audio_x")
    # Speak the text
    engine.save_to_file(text, audiofile+".wav")
    engine.runAndWait()

    # Load your WAV audio file
    audio_file_path = audiofile+".wav"

    # Read file in binary mode
    with open(audio_file_path, "rb") as file:
        audio_bytes = file.read()

    # Download button for the WAV file
    st.download_button(
        label="📥 Download WAV Audio",
        data=audio_bytes,
        file_name=audio_file_path,
        mime="audio/wav"
    )

    # Optional: Stream the audio in the app
    st.audio(audio_bytes, format="audio/wav")

def Local_Machine_Learning():
    import joblib

    vectorizer = joblib.load('tfidf_vectorizer_fin.pkl')
    model = joblib.load('emotion_model_xgb.pkl')

    new_text=open("output.txt","r").read()
    
    st.markdown("## User: \n  ###\t "+new_text+"\n\n")
    new_vec = vectorizer.transform([new_text])
    
    label_map = {
        0: 'anger',        # from reference (3)
        1: 'anxiety',
        2: 'calm',
        3: 'confusion',
        4: 'desperation',
        5: 'fear',         # from reference (4)
        6: 'frustration',
        7: 'joy',          # from reference (1)
        8: 'neutral',
        9: 'panic',
        10: 'sadness',     # from reference (0)
        11: 'shock',
        12: 'tension'
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

    def Speech_Recognizer():
        def prepare_voice_file(path: str) -> str:
            """
            Converts audio to WAV if needed. Whisper supports multiple formats, but WAV is a safe choice.
            """
            if os.path.splitext(path)[1] == '.wav':
                return path
            elif os.path.splitext(path)[1] in ('.mp3', '.m4a', '.ogg', '.flac'):
                audio_file = AudioSegment.from_file(path, format=os.path.splitext(path)[1][1:])
                wav_file = os.path.splitext(path)[0] + '.wav'
                audio_file.export(wav_file, format='wav')
                return wav_file
            else:
                raise ValueError(f'Unsupported audio format: {os.path.splitext(path)[1]}')

        def transcribe_with_whisper(input_path: str):
            model = whisper.load_model("base")  # or "small", "medium", "large"
            
            # Load and preprocess audio
            audio = whisper.load_audio(input_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # Language detection
            _, probs = model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)

            st.markdown(f"**Detected Language:** `{detected_lang}`")

            # Transcription
            result_transcribe = model.transcribe(input_path, fp16=False)
            st.markdown("### 🔊 Transcription:")
            st.write(result_transcribe["text"])

            # Translation to English
            result_translate = model.transcribe(input_path, task="translate", fp16=False)
            st.markdown("### 🌐 Translated to English:")
            st.write(result_translate["text"])
            with open("output.txt", 'w') as f:
                f.write(result_translate["text"])
            st.write("Text Saved.")

        # Streamlit UI
        st.title(" Language Detection")

        uploaded_files = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a", "flac", "ogg"], accept_multiple_files=True)

        for uploaded_file in uploaded_files:
            st.write(f"**File Uploaded:** `{uploaded_file.name}`")
            st.audio(uploaded_file, format="audio/mpeg", loop=False)

            # Save uploaded file
            input_path = uploaded_file.name
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Prepare and run transcription
            try:
                wav_file = prepare_voice_file(input_path)
                transcribe_with_whisper(wav_file)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    Speech_Recognizer()

def Transcription_Translation():
    import streamlit as st
    import whisper
    import os
    from pydub import AudioSegment

    def Speech_Recognizer():
        def prepare_voice_file(path: str) -> str:
            """
            Converts audio to WAV if needed. Whisper supports multiple formats, but WAV is a safe choice.
            """
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
            model = whisper.load_model("medium")  # Use "medium" or "large" for better regional support

            audio = whisper.load_audio(input_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            _, probs = model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            st.markdown(f"🔍 **Detected Language:** `{detected_lang}`")

            # Transcription
            transcription = model.transcribe(input_path, language=detected_lang, fp16=False)
            st.markdown("### 📝 Transcription in Native Language:")
            st.write(transcription["text"])

            # Translation (only for ta/te)
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
        # Streamlit UI
        st.title("🎙️ Language Detection & Translation")

        uploaded_files = st.file_uploader("📤 Upload audio file", type=["mp3", "wav", "m4a", "flac", "ogg"], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.write(f"**File Uploaded:** `{uploaded_file.name}`")
                st.audio(uploaded_file, format="audio/mpeg", loop=False)

                input_path = uploaded_file.name
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    wav_path = prepare_voice_file(input_path)
                    transcribe_and_translate(wav_path)
                    st.markdown("---")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    Speech_Recognizer()


page_names_to_funcs = {
    "Transcription Translation":Transcription_Translation,
    "Language Detection":Language_Detection,
    "Local ML": Local_Machine_Learning,
    "Text to Speech": TTS,
    #"Speech Recognition": Speech_Recognizer,
    "Emotion Analyzer":Emotion_Analyzer,
    "Home": Home,
}
st.sidebar.markdown("# EMOLYZER")
st.sidebar.image("./emo.png")
demo_name = st.sidebar.selectbox("What you want to do?", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()