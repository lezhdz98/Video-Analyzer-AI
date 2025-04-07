import streamlit as st
import requests
import os

# API endpoints
TRANSCRIBE_URL = "http://localhost:5000/transcribe"
DETAILED_SUMMARY_URL = "http://localhost:5000/summarize_detailed"
CONCISE_SUMMARY_URL = "http://localhost:5000/summarize_concise"

# Streamlit UI
st.title("🎥 Video Analyzer")

uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# Reset session state if a different video is uploaded
if uploaded_video is not None:
    if "last_uploaded_filename" in st.session_state and st.session_state["last_uploaded_filename"] != uploaded_video.name:
        for key in ["transcript", "detailed_summary", "concise_summary"]:
            st.session_state.pop(key, None)

    st.session_state["last_uploaded_filename"] = uploaded_video.name

    video_path = os.path.join("uploads", uploaded_video.name)

    # Save the uploaded video to disk
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.session_state["video_path"] = video_path  # Track video path in session state

# Show video only if it's stored in session
if "video_path" in st.session_state:
    st.video(st.session_state["video_path"])

    if st.button("Transcribe Video"):
        with st.spinner("Transcribing..."):
            files = {'video': open(st.session_state["video_path"], 'rb')}
            response = requests.post(TRANSCRIBE_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                transcript = result["transcript"]
                st.session_state["transcript"] = transcript
                st.success("✅ Transcription complete!")

                # Show transcript in expander
                with st.expander("📄 Show Transcript"):
                    st.write(transcript)
            else:
                st.error("❌ Transcription failed.")

# Summarization Tabs
if "transcript" in st.session_state:
    transcript = st.session_state["transcript"]
    tab1, tab2 = st.tabs(["📝 Detailed Summary", "🧠 Concise Summary (100 words)"])

    with tab1:
        if "detailed_summary" not in st.session_state:
            with st.spinner("Generating detailed summary..."):
                response = requests.post(DETAILED_SUMMARY_URL, json={"transcript": transcript})
                if response.status_code == 200:
                    st.session_state["detailed_summary"] = response.json()["analysis"]
                else:
                    st.session_state["detailed_summary"] = "⚠️ Error generating detailed summary."

        st.write(st.session_state["detailed_summary"])

    with tab2:
        if "concise_summary" not in st.session_state:
            with st.spinner("Generating concise summary..."):
                response = requests.post(CONCISE_SUMMARY_URL, json={"transcript": transcript})
                if response.status_code == 200:
                    st.session_state["concise_summary"] = response.json()["summary"]
                else:
                    st.session_state["concise_summary"] = "⚠️ Error generating concise summary."

        st.write(st.session_state["concise_summary"])

# Separator and Reset Button
st.markdown("---")

# 🔄 Big Reset Button
if st.button("🔁 Reset All"):
    for key in ["transcript", "detailed_summary", "concise_summary", "last_uploaded_filename", "video_path"]:
        st.session_state.pop(key, None)
    st.rerun()
