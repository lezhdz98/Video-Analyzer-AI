import whisper

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result['text']

# Example usage
if __name__ == "__main__":
    audio_file = "output_audio.mp3"
    transcription = transcribe_audio(audio_file)
    print("Transcription:", transcription)
