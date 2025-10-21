# AI-Powered Video Analyzer

An intelligent video analysis app that automatically transcribes videos, summarizes the transcript, and describes key visual scenes using state-of-the-art language and vision models.  

Built with **Flask** (backend) and **Streamlit** (frontend), this tool leverages **FFmpeg**, **Whisper**, **LangChain**, and **OpenAI** models to deliver rich, multi-modal insights from videos.

---

## Overview

This app enables users to upload videos and receive:

- A **transcript** of the audio using Whisper.
- A **summary** of the video content (detailed or concise).
- **Scene descriptions** generated from key frames using vision-language models.
- **Topic tags** extracted from the video content.
- A **holistic summary** combining text and visual analysis.
- Support for **multiple languages** in transcription and summarization.

---

## Features

### Transcription
- Extracts audio from video using `ffmpeg`.
- Converts audio to text using Whisper.

### Summary Generation
- Summarizes the transcript using OpenAI models.
- Supports:
  - Concise summaries (100 words)
  - Detailed summaries

### Scene Description
- Extracts keyframes using `ffmpeg`.
- Describes each scene using OpenAI Vision analysis.

### Multi-Modal Integration
- Combines text and image descriptions for holistic understanding.
- Example output:  
  _"The video is a lecture on AI with slides discussing neural networks."_

### Topic Tagging
- Automatically generates relevant tags from transcript and visuals.

### Interactive Interface
- Streamlit-based web interface to:
  - Upload videos
  - View transcripts, summaries, descriptions, and tags

### Multilingual Support
- Transcribes and summarizes content in various languages.

---

## Prerequisites

- **Python**: 3.12.7  
- **FFmpeg**: Required for audio and frame extraction

### Install FFmpeg (macOS)
```bash
    brew install ffmpeg
```

### Install FFmpeg (Linux)
```bash
   sudo apt update
   sudo apt install ffmpeg
```

### Install FFmpeg (Windows)
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html).
2. Extract the ZIP archive to a folder, e.g., `C:\ffmpeg`.
3. Add the `bin` folder to your system PATH:
   - Search for **Environment Variables** in the Start Menu.
   - Under **System variables**, find `Path`, then click **Edit**.
   - Click **New** and add: `C:\ffmpeg\bin`
4. Open a new command prompt and run `ffmpeg -version` to verify the installation.

---

## API Keys & Environment Variables

Set the following in a `.env` file or your system environment:

```bash
    OPENAI_API_KEY=your-openai-api-key

    # Optional: LangSmith integration for tracing/debugging
    LANGSMITH_TRACING=true
    LANGSMITH_ENDPOINT=
    LANGSMITH_API_KEY=
    LANGSMITH_PROJECT=
```

---

## Environment Setup

1. **Clone the repository:**
```bash
   git clone https://github.com/your-username/ai-video-analyzer.git
   cd ai-video-analyzer
```

2. **Create a virtual environment:**
```bash
   python3 -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

---

## Run the App

1. **Start the backend server (Flask):**
```bash
   python server.py
```

2. **Launch the frontend (Streamlit):**
```bash
   streamlit run frontend.py
```
3. **View the frontend in your browser:**
   - Go to [http://localhost:8501](http://localhost:8501) by default.

---

## Project Structure 

```bash
    .
    ├── server.py                   # Flask backend
    ├── frontend.py                 # Streamlit UI
    ├── video_audio_extractor.py    # Audio extraction
    ├── frame_extractor.py          # Frame extraction using FFmpeg
    ├── transcribe_audio.py         # Transcrib extraction using Whisper
    ├── requirements.txt
    └── .env                        # API keys and config
```

## THIS SECTION WAS MADE FROM AI FOUNDRY