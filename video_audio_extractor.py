import ffmpeg

def extract_audio(video_file, audio_file):
    try:
        # Extract audio from the video file
        ffmpeg.input(video_file).output(audio_file).run()
        print(f"Audio extracted successfully to {audio_file}")
    except ffmpeg.Error as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    video_file = "input_video.mp4"
    audio_file = "output_audio.mp3"
    extract_audio(video_file, audio_file)
