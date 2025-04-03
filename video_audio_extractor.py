import ffmpeg

def extract_audio(video_file, audio_file):
    try:
        # Extract audio from the video file
        ffmpeg.input(video_file).output(audio_file).run()
        print(f"Audio extracted successfully to {audio_file}")
    except ffmpeg.Error as e:
        print(f"An error occurred: {e}")

# Example usage
video_file = 'friends_phonebill.mp4'  
audio_file = 'output_audio.mp3'  

extract_audio(video_file, audio_file)
