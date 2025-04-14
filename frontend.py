import streamlit as st
import requests
import os
import base64

# API endpoints
TRANSCRIBE_URL = "http://localhost:5000/transcribe"
SUMMARY_URL = "http://localhost:5000/summarize_custom"
TAGS_URL = "http://localhost:5000/generate_tags"
FRAMES_URL = "http://localhost:5000/frames_description"
HOLISTIC_SUMMARY_URL = "http://localhost:5000/holistic_summary"

def send_post_request(url, payload=None, files=None):
    """
    Helper function to send a POST request to the given URL.

    Args:
        url (str): The API endpoint.
        payload (dict, optional): JSON payload for the request.
        files (dict, optional): Files to upload (used for transcription).

    Returns:
        Response object or None if an error occurs.
    """
    try:
        if files:
            response = requests.post(url, files=files)
        else:
            response = requests.post(url, json=payload)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Request to {url} failed: {e}")
        return None

# Set up the layout for the Streamlit app
st.title("🎥 AI Video Analyzer")
st.markdown("#### Upload a video to get a custom summary, transcription, and key frames.")

# Upload video file
uploaded_video = st.file_uploader("📤 Choose a video file", type=["mp4", "avi", "mov"])

# Initialize session state if it's not already initialized
if "last_uploaded_filename" not in st.session_state:
    st.session_state["last_uploaded_filename"] = ""

# Handle new video uploads
if uploaded_video is not None:
    if st.session_state["last_uploaded_filename"] != uploaded_video.name:
        for key in ["transcript", "custom_summary", "frames", "tags"]:
            st.session_state.pop(key, None)

    st.session_state["last_uploaded_filename"] = uploaded_video.name
    video_path = os.path.join("uploads", uploaded_video.name)

    # Save video
    try:
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
    except Exception as e:
        st.error(f"Error saving uploaded video: {e}")
        st.stop()

    st.session_state["video_path"] = video_path  # Track video path in session state
    st.video(video_path)

    # --- Summary Form ---
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

    if submitted:
        # Transcription
        with st.spinner("⏳ Transcribing video..."):
            files = {'video': open(st.session_state["video_path"], 'rb')}
            transcribe_response = send_post_request(TRANSCRIBE_URL, files=files)

        if transcribe_response:
            transcript = transcribe_response.json().get("transcript", "")
            st.session_state["transcript"] = transcript
        else:
            st.error("❌ Transcript generation failed.")
            st.stop()

        # Custom summary
        with st.spinner("📝 Generating custom summary..."):
            payload = {
                "transcript": transcript,
                "summary_type": summary_type.lower(),
                "language": language,
                "style": style
            }
            summary_response = send_post_request(SUMMARY_URL, payload=payload)

        if summary_response:
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

        # Generate Tags
        with st.spinner("🏷️ Generating tags..."):
            tags_payload = {"transcript": transcript}
            tags_response = send_post_request(TAGS_URL, payload=tags_payload)

        if tags_response:
            tags = tags_response.json().get("tags", [])
            st.session_state["tags"] = tags
            st.success("✅ Tags generated!")
            st.markdown("### 🏷️ Relevant Tags")

            # Display tags with "#" before each tag
            hashtagged_tags = [f"#{tag.replace(' ', '')}" for tag in tags]
            st.write(" ".join(hashtagged_tags))
        else:
            st.error("❌ Tag generation failed.")
            st.stop()

        # Extract and Analyze Frames
        with st.spinner("🖼️ Extracting frames..."):
            frames_payload = {
                "video_path": st.session_state["video_path"],
                "language": language  # Include selected language from summary form
            }
            frames_response = send_post_request(FRAMES_URL, payload=frames_payload)

        if frames_response:
            frames = frames_response.json().get("frames", [])
            st.session_state["frames"] = frames

            if frames:
                st.markdown("### 🖼️ Key Frames")
                with st.expander("📄 Show Frames"):
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
            st.stop()

        # Holistic Summary
        if "transcript" in st.session_state and "frames" in st.session_state:
            st.markdown("### 📘 Holistic Summary (textual + visual analysis)")

            with st.spinner("Creating a holistic summary combining textual and visual analysis..."):
                holistic_payload = {
                    "transcript": st.session_state["transcript"],
                    "frames": st.session_state["frames"],
                    "language": language
                }
                holistic_response = send_post_request(HOLISTIC_SUMMARY_URL, payload=holistic_payload)

            if holistic_response:
                holistic_summary = holistic_response.json().get("holistic_summary", "")
                st.success("✅ Holistic summary generated!")

                with st.expander("📄 Show Holistic Summary"):
                    st.write(holistic_summary)
            else:
                st.error("❌ Holistic summary generation failed.")
                st.stop()
