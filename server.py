from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from video_audio_extractor import extract_audio
from transcribe_audio import transcribe_audio
from frame_extractor import extract_frames
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

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize the chat model with the API key
model = init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=api_key)

# Initialize the vision model with the API key
#model_vision = init_chat_model( model="gpt-4o-mini", model_provider="openai", openai_api_key=api_key)

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
def analyze_transcript(transcript, prompt_template):
    prompt = prompt_template.format_messages(transcript=transcript)
    response = model.invoke(prompt)
    return response.content

def get_frames(video_path):
    # Create directory for storing frames (inside the 'frames' folder directly)
    os.makedirs("frames", exist_ok=True)

    # Extract frames
    extract_frames(video_path, "frames", num_frames=10)

    frames = []

    # Read and encode each frame as base64
    for filename in sorted(os.listdir("frames")):
        print(f"Processing file: {filename}")
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Check if it's a frame image
            with open(os.path.join("frames", filename), "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                frames.append({"name": filename, "image": encoded})

    print(f"Extracted {len(frames)} frames from {video_path}")

    return frames

def analyze_frames(frames):
    # Step 1: Define the schema for structured output
    schema = {
        "title": "DescriptionFrames",
        "description": "Frames description with vision",
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Image file name"
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the visual content"
                        }
                    },
                    "required": ["image", "description"]
                },
                "description": "A list of frame descriptions"
            }
        },
        "required": ["frames"]
    }

    # Step 2: Wrap the model to produce structured output
    structured_lm = model.with_structured_output(schema)

    # Step 3: Prompt template (multi-modal)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that describes the content of each image frame. Please provide a description in less than 20 words."),
        ("user", "{image_prompt}")
    ])

    chain = prompt_template | structured_lm

    # Step 4: Analyze each frame individually and collect results
    results = []
    for frame in frames:
        base64_image = frame["image"]
        image_name = frame["name"]

        # Debug: Check if the base64 image is different for each frame
        #print(f"Base64 for {image_name}: {base64_image[:50]}...")  # Print a part of the base64 for comparison
        #print len(base64_image)
        print(f"Analyzing frame: {image_name}")
        print(f"Base64 length: {len(base64_image)}")

        image_prompt = [
            {
                "type": "text", 
                "text": f'''
                            Describe this frame: {image_name}.
                            Focus on the key elements that make this frame unique. 
                            What specific objects, colors, and features stand out in this scene? 
                            What mood or atmosphere does this frame convey? 
                            Describe the details of the scene, such as people, animals, nature, objects, weather, or any movement if visible.
                        '''
            },
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]

        # Invoke the vision model
        response = chain.invoke({"image_prompt": image_prompt})


        print(f"Response for {image_name}: {response}")
        # Append the result
        results.append({
            "image": image_name,
            "description": response.get("frames", [{}])[0].get("description", "No description found.")
        })

    # Step 5: Return full structured result
    print("Structured result:", results)
    print(f"Analyzed {len(results)} frames.")
    return {"frames": results}

def holistic_summary(transcript, frames_analysis):
    
    # Define the holistic summary prompt
    system_message = '''
        You are a helpful assistant that generates a holistic summary based on video content.
        The user will provide a transcript and frames from a video, and you will summarize it.
        Please write a summary of the given transcript and frames.
    '''
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Transcript: {transcript} Frames: {frames}")
    ])
    
    result = analyze_transcript(transcript, prompt_template)
    return result

# Route for transcribing video and generating summary
@app.route('/transcribe', methods=['POST'])
def transcribe():
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

# Route for custom summary generation
@app.route('/summarize_custom', methods=['POST'])
def summarize_custom():
    data = request.get_json()
    transcript = data.get("transcript")
    summary_type = data.get("summary_type", "Detailed")
    language = data.get("language", "English")
    style = data.get("style", "Formal")

    if not transcript:
        return jsonify({"error": "Transcript is required"}), 400

    if summary_type.lower() == "detailed":
        summary_template = "Detailed summary with key points."
    else:
        summary_template = "Concise summary in 100 words."

    # Generate system message for custom summary based on input
    system_message = f'''
        You are a helpful assistant that generates summaries based on user input.
        The user will provide a transcript of a video, and you will summarize it.
        Please write a summary of the given transcript with the following characteristics:
        - Style: {style}
        - Summary Type: {summary_template}
        - Language: {language}
    '''
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{transcript}")
    ])

    result = analyze_transcript(transcript, prompt_template)
    return jsonify({"summary": result}), 200

# Route for generating tags from transcript
@app.route('/generate_tags', methods=['POST'])
def generate_tags():
    data = request.get_json()
    transcript = data.get("transcript")
    if not transcript:
        return jsonify({"error": "Transcript is required"}), 400

    # Step 1: Define Schema
    schema = {
        "title": "TranscriptTags",
        "description": "Extracted tags from a transcript.",
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {
                    "type": "string",
                },
                "description": "A list of relevant keywords or tags from the transcript.",
            },
        },
        "required": ["tags"]
    }

    # Step 2: Wrap model with structured output
    structured_llm = model.with_structured_output(schema)

    # Step 3: Prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Extract 5 to 10 relevant tags (keywords) from the transcript."),
        ("user", "{transcript}")
    ])
    
    # Step 4: Build and invoke chain
    chain = prompt_template | structured_llm 
    result = chain.invoke({"transcript": transcript})

    # Step 5: Return clean JSON
    return jsonify(result), 200

# Route for extracting frames and analyzing them
@app.route('/frames_description', methods=['POST'])
def frames_analysis():
    data = request.get_json()
    video_path = data.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Invalid video path"}), 400

    # Step 1: Extract + encode frames
    frames = get_frames(video_path)

    # Step 2: Analyze frames
    analysis = analyze_frames(frames) 

    # Step 4: Merge base64 back into results
    for i, frame in enumerate(analysis["frames"]):
        frame["image"] = frames[i]["image"]  # add the base64 back for Streamlit to render

    return jsonify(analysis)

if __name__ == '__main__':
    app.run(debug=True)