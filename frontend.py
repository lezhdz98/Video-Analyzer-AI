import streamlit as st
import requests
import os
import base64

# API endpoints
TRANSCRIBE_URL = "http://localhost:5000/transcribe"
SUMMARY_URL = "http://localhost:5000/summarize_custom"
TAGS_URL = "http://localhost:5000/generate_tags"
FRAMES_URL = "http://localhost:5000/frames_description"

# Set up the layout for the Streamlit app
st.title("🎥 AI Video Analyzer")
st.markdown("#### Upload a video to get a custom summary, transcription, and key frames.")

# Upload video file
uploaded_video = st.file_uploader("📤 Choose a video file", type=["mp4", "avi", "mov"])

# Initialize session state if it's not already initialized
if "last_uploaded_filename" not in st.session_state:
    st.session_state["last_uploaded_filename"] = ""

# Reset session state if a different video is uploaded
if uploaded_video is not None:
    if st.session_state["last_uploaded_filename"] != uploaded_video.name:
        for key in ["transcript", "custom_summary", "frames"]:
            st.session_state.pop(key, None)

    st.session_state["last_uploaded_filename"] = uploaded_video.name

    video_path = os.path.join("uploads", uploaded_video.name)

    # Save the uploaded video to disk
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.session_state["video_path"] = video_path  # Track video path in session state
    st.video(video_path)

    # -- Step 1: Show the summary form
    st.markdown("### 🧠 Customize Your Summary")

    with st.form("summary_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            summary_type = st.radio("Summary Type", ["Detailed", "Concise"], horizontal=True)

        with col2:
            language = st.selectbox("Language", ["English", "French", "Spanish"])

        with col3:
            style = st.selectbox("Style", ["Formal", "Informal"])

        submitted = st.form_submit_button("⚙️ Analyze Video")

    # -- Step 2: Process the video 
    if submitted:
        # -- Transcribe the video
        with st.spinner("⏳ Transcribing video..."):
            files = {'video': open(st.session_state["video_path"], 'rb')}
            transcribe_response = requests.post(TRANSCRIBE_URL, files=files)

        if transcribe_response.status_code == 200:
            transcript = transcribe_response.json().get("transcript", "")
            st.session_state["transcript"] = transcript
        else:
            st.error("❌ Transcription failed.")
            st.stop()

        # -- Generate custom summary
        with st.spinner("📝 Generating custom summary..."):
            payload = {
                "transcript": transcript,
                "summary_type": summary_type.lower(),
                "language": language,
                "style": style
            }
            summary_response = requests.post(SUMMARY_URL, json=payload)

        if summary_response.status_code == 200:
            summary = summary_response.json().get("summary", "")
            st.session_state["custom_summary"] = summary
            st.success("✅ Summary generated!")
            st.markdown("### 📋 Custom Summary")
            st.write(summary)
        else:
            st.error("❌ Summary generation failed.")
            st.stop()

        with st.expander("📄 Show Transcript"):
            st.write(transcript)

        # -- Generate tags from the transcript 
        with st.spinner("🏷️ Generating tags..."):
            tags_payload = {"transcript": transcript}
            tags_response = requests.post(TAGS_URL, json=tags_payload)

        if tags_response.status_code == 200:
            tags = tags_response.json().get("tags", [])
            st.session_state["tags"] = tags
            st.success("✅ Tags generated!")
            st.markdown("### 🏷️ Relevant Tags")

            # Display tags with "#" before each tag
            hashtagged_tags = [f"#{tag.replace(' ', '')}" for tag in tags]
            st.write(" ".join(hashtagged_tags))  # Join tags into a single string with spaces
        else:
            st.error("❌ Tag generation failed.")
            st.stop()

        # -- Extract frames from the video
        # -- Extract frames from the video
        with st.spinner("🖼️ Extracting frames..."):
            frames_payload = {
                "video_path": st.session_state["video_path"],
                "language": language  #include selected language from summary form
            }
            frames_response = requests.post(FRAMES_URL, json=frames_payload)

        if frames_response.status_code == 200:
            frames = frames_response.json().get("frames", [])
            st.session_state["frames"] = frames

            if frames:
                st.markdown("### 🖼️ Key Frames")

                for frame in frames:
                    # Decode the base64 image string and pass it to st.image
                    image_data = base64.b64decode(frame["image"])

                    # Get the description for the frame (from the backend analysis)
                    description = frame.get("description", "No description available.")

                    # Display the frame with its description
                    st.image(image_data, caption=frame.get("name"), use_container_width=True)
                    st.write(f"**Description**: {description}")
            else:
                st.info("No frames returned.")
        else:
            st.error("❌ Frame extraction failed.")
