import whisper

def transcribe_audio(audio_file):
    """
    Transcribe speech from an audio file using OpenAI's Whisper model.

    Parameters:
    - audio_file: path to the audio file to be transcribed

    Returns:
    - The transcribed text as a string

    Notes:
    - Uses the 'base' Whisper model.
    - Supports multiple audio formats (e.g., mp3, wav, m4a).
    """
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        return result['text']
    except FileNotFoundError:
        return f"Error: The file '{audio_file}' was not found."
    except Exception as e:
        return f"An error occurred during transcription: {e}"

# Example usage
if __name__ == "__main__":
    audio_file = "output_audio.mp3"
    transcription = transcribe_audio(audio_file)
    print("Transcription:", transcription)
