from flask import Flask, request, jsonify
import os
from video_audio_extractor import extract_audio
from transcribe_audio import transcribe_audio
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from werkzeug.utils import secure_filename

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key is not set in the environment variables.")

# Optional: print the API key to verify it's loaded correctly
print(f"OpenAI API Key: {api_key}")

# Initialize Flask app
app = Flask(__name__)

# Initialize the chat model with the API key
model = init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=api_key)

# Define system template for video analysis
system_template = "Analyze the following video transcript and summarize the key points."

# Define user template for transcript
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{transcript}")]
)


# Folder to save uploaded videos and extracted audio
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'audio_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Configure Flask to allow file uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to analyze the video transcript using the LLM (OpenAI)
def analyze_video(transcript):
    # Format the messages using the transcript
    prompt = prompt_template.format_messages(transcript=transcript)
    # Send messages to the model
    response = model.invoke(prompt)
    return response.content


@app.route('/transcribe_and_analyze_video', methods=['POST'])
def transcribe_and_analyze_video():
    # Check if the video file is part of the request
    if 'video' not in request.files:
        return jsonify({"error": "No video file found in request"}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if video_file and allowed_file(video_file.filename):
        # Save the uploaded video file to disk
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
        video_file.save(video_path)

        # Extract audio from the video
        audio_path = os.path.join(AUDIO_FOLDER, f"{os.path.splitext(video_file.filename)[0]}.mp3")
        extract_audio(video_path, audio_path)

        # Transcribe the audio
        transcript = transcribe_audio(audio_path)

        print(f"Transcript: {transcript}")
        
        # Call the analyze_video function to process the transcript with the LLM
        analysis = analyze_video(transcript)
        
        return jsonify({"transcript": transcript, "analysis": analysis}), 200

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)