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

# Detailed Summary System Prompt
detailed_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Analyze the following video transcript and summarize the key points."),
    ("user", "{transcript}")
])

# Concise Summary System Prompt
concise_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following transcript in 100 words."),
    ("user", "{transcript}")
])

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

# Function to analyze video transcript
def generate_summary(transcript, prompt_template):
    prompt = prompt_template.format_messages(transcript=transcript)
    response = model.invoke(prompt)
    return response.content

# Route for transcribing video and generating summary
@app.route('/transcribe', methods=['POST'])
def transcribe_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    
    if video_file.filename == '' or not allowed_file(video_file.filename):
        return jsonify({"error": "Invalid file"}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
    video_file.save(video_path)
    
    audio_path = os.path.join(AUDIO_FOLDER, f"{os.path.splitext(video_file.filename)[0]}.mp3")
    extract_audio(video_path, audio_path)
    transcript = transcribe_audio(audio_path)

    return jsonify({"transcript": transcript}), 200

# Route for detailed summary
@app.route('/summarize_detailed', methods=['POST'])
def summarize_detailed():
    data = request.get_json()
    transcript = data.get("transcript")
    if not transcript:
        return jsonify({"error": "Transcript is required"}), 400

    analysis = generate_summary(transcript, detailed_prompt_template)
    return jsonify({"analysis": analysis}), 200

# Route for concise summary
@app.route('/summarize_concise', methods=['POST'])
def summarize_concise():
    data = request.get_json()
    transcript = data.get("transcript")
    if not transcript:
        return jsonify({"error": "Transcript is required"}), 400

    summary = generate_summary(transcript, concise_prompt_template)
    return jsonify({"summary": summary}), 200

if __name__ == '__main__':
    app.run(debug=True)