import streamlit as st
import requests
import os

# Define the API endpoint
API_URL = "http://localhost:5000/transcribe_and_analyze_video"

# Streamlit UI for video upload
st.title("Video Analyzer")
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    video_path = os.path.join("uploads", uploaded_video.name)
    
    # Save the uploaded video to disk
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(video_path)  # Display the video
    
    # Call the backend API to process the video
    if st.button("Analyze Video"):
        with st.spinner("Processing..."):
            # Send the video to the backend API for analysis
            files = {'video': open(video_path, 'rb')}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                result = response.json()  # Get both transcript and analysis
                st.subheader("Transcript")
                st.write(result["transcript"])
                st.subheader("Analysis")
                st.write(result["analysis"])
            else:
                st.error("Error during video analysis.")
