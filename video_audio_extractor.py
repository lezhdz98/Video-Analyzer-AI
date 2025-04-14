import ffmpeg
import os

def extract_audio(video_file, audio_file):
    """
    Extract audio from a video file using ffmpeg.

    Parameters:
    - video_file: path to the input video file
    - audio_file: path where the extracted audio will be saved

    Notes:
    - Overwrites the audio file if it already exists.
    - Prints a success message or error details.
    """
    try:
        # Remove existing audio file if it exists to avoid overwrite prompt
        if os.path.exists(audio_file):
            os.remove(audio_file)

        # Extract audio from the video file, with overwrite enabled
        ffmpeg.input(video_file).output(audio_file).run(overwrite_output=True)
        print(f"Audio extracted successfully to {audio_file}")
    except ffmpeg.Error as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    video_file = "input_video.mp4"
    audio_file = "output_audio.mp3"
    extract_audio(video_file, audio_file)