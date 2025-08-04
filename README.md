
# 🎧 Emotion-Aware Call Analysis System for Cybercrime Incident Assessment

A system designed to analyze caller emotions from audio and transcripts to support cybercrime incident assessment. The goal is to help law enforcement teams assess the emotional urgency of reports by detecting cues such as **fear, anger, panic, or distress**.

---

## 📌 Objective

This system processes **recorded or ingested audio calls** and applies emotion analysis techniques to identify emotional states. These insights can support response prioritization and help officers focus on emotionally urgent cases.

---

## ✅ Scope

* Audio ingestion and **speech-to-text conversion**
* Emotion classification using **speech features** and **text transcripts**
* Tagging conversations with **emotional urgency levels**
* Officer dashboard to **review and filter flagged conversations**

> ⚠️ **Note:** This system currently operates on uploaded or recorded call data. Real-time streaming is not yet implemented.

---

## 🧠 Architecture Overview

1. **Audio Capture & Ingestion**
   Accept audio files or recordings for analysis.

2. **Preprocessing & Feature Extraction**

   * Clean and segment audio
   * Extract speech features (e.g., MFCCs, pitch)
   * Convert speech to text using ASR

3. **Emotion Detection (Audio + Text)**

   * Analyze emotions using classifiers on both audio and text
   * Combine predictions for more accurate emotion detection

4. **Urgency Classifier**

   * Map emotions to urgency levels (e.g., panic → high urgency)

5. **Visualization & Alert Management**

   * View flagged conversations in a Streamlit-based dashboard
   * Filter by emotion or urgency level

---

## ⚙️ How It Works

1. Upload a call recording (e.g., `.wav`, `.mp3`)
2. Audio is processed and transcribed
3. Emotion detection models analyze both audio and transcript
4. Emotion tags are mapped to urgency categories
5. Results are shown in a dashboard for review and decision-making

---

## 🧪 Tech Stack

* **Python**
* **PyTorch / TensorFlow**
* **SpeechRecognition / Whisper / Vosk**
* **Streamlit**
* **Transformers / Scikit-learn**

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```
.
├── app.py                       # Streamlit dashboard
├── audio_ingestion.py          # Load and process audio input
├── preprocessing.py            # Speech feature + text extraction
├── emotion_detection.py        # Emotion classification logic
├── urgency_classifier.py       # Map emotions to urgency tags
├── dashboard/                  # UI and alert views
├── models/                     # Trained models and configs
└── requirements.txt
```

---

## 🧭 Future Work

* 🔄 Real-time audio streaming support
* 🧠 Improved emotion fusion across modalities
* 🛠️ Feedback loop for officer-based label refinement

---
"# Emolyzer" 
