"""
    Flask app for AI-powered video analysis.

    This application provides several endpoints to:
    - Transcribe video content
    - Generate custom summaries and tags from transcripts
    - Extract image frames and generate AI-based visual descriptions
    - Generate a holistic summary combining both transcript and visual analysis

    Tech stack:
    - Flask for API backend
    - LangChain + OpenAI for NLP tasks
    - Whisper for audio transcription
    - Custom utilities for video/audio/frame extraction
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import json
from video_audio_extractor import extract_audio
from transcribe_audio import transcribe_audio
from frame_extractor import extract_frames
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langsmith.run_helpers import traceable
from langchain_openai import ChatOpenAI
from werkzeug.utils import secure_filename
from openai import OpenAI


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
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to analyze video transcript
def analyze_transcript(transcript, prompt_template):
    """Analyze a transcript using a given LangChain prompt template."""
    try:
        prompt = prompt_template.format_messages(transcript=transcript)
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error analyzing transcript: {e}")
        return "Error generating summary."

def get_frames(video_path):
    """
    Extract 10 frames from a video and encode them in base64 for OpenAI analysis.
    Returns a list of dictionaries with frame names and base64 strings.
    """
    try:
        os.makedirs("frames", exist_ok=True)
        extract_frames(video_path, "frames", num_frames=10)
        frames = []

        # Read and encode each frame as base64
        for filename in sorted(os.listdir("frames")):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                with open(os.path.join("frames", filename), "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode('utf-8')
                    frames.append({"name": filename, "image": encoded})

        print(f"Extracted {len(frames)} frames from {video_path}")

        return frames
        
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return []

#langsmith trace
@traceable(name="OpenAI Frame Description")
def analysis_with_openAI_vanilla(frames, language="English"):
    """
    Analyze video frames using OpenAI's tool-calling to generate descriptions.
    Returns a list of descriptions associated with each frame.
    """
    print("Using OpenAI for frame analysis...")

    client = OpenAI(api_key=api_key)
    results = []

    tools = [
        {
            "type": "function",
            "function": {
                "name": "describe_image",
                "description": "Generate a description for an image frame.",
                "parameters": {
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
                    "required": ["image", "description"],
                    "additionalProperties": False
                }
            }
        }
    ]

    for frame in frames:
        image_name = frame["name"]
        base64_image = frame["image"]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that describes the content of each image frame in the given language: {language}. You always responds using the 'describe_image' tool."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", 
                            "text": f'''
                                    Describe this image frame: {image_name}.
                                    Language of the description: {language}.
                                    Describe it in less than 50 words.
                                    Focus on the key elements that make this frame unique. 
                                    What specific objects, colors, and features stand out in this scene? 
                                    What mood or atmosphere does this frame convey? 
                                    Describe the details of the scene, such as people, animals, nature, objects, weather, or any movement if visible.
                                    '''
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ]
                    }
                ],
                tools=tools,
                tool_choice="auto"
            )

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and len(tool_calls) > 0:
                tool_response = tool_calls[0].function.arguments
                tool_data = json.loads(tool_response)

                results.append({
                    "image_name": tool_data["image"],
                    "description": tool_data["description"]
                })
            else:
                print(f"No tool call detected for {image_name}.")
                results.append({
                    "image_name": image_name,
                    "description": "Tool call was not triggered."
                })

        except Exception as e:
            print(f"Error analyzing frame {image_name}: {e}")
            results.append({
                "image_name": image_name,
                "description": "Error generating description."
            })

    return results

# Route for transcribing video and generating summary
@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Upload a video file, extract audio, and return transcript.
    """
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']

        if video_file.filename == '' or not allowed_file(video_file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
        video_file.save(video_path)

        audio_path = os.path.join(AUDIO_FOLDER, f"{os.path.splitext(video_file.filename)[0]}.mp3")

        try:
            extract_audio(video_path, audio_path)
            transcript = transcribe_audio(audio_path)
        except Exception as e:
            print(f"Error analyzing transcript: {e}")
            return jsonify({"error": f"Failed to process video: {str(e)}"}), 500

        return jsonify({"transcript": transcript}), 200
    except Exception as e:
        print(f"Failed to process video: {e}")
        return jsonify({"error": f"Failed to process video: {e}"}), 500

# Route for custom summary generation
@app.route('/summarize_custom', methods=['POST'])
def summarize_custom():
    """
    Generate a custom summary based on user-defined style, language, and detail level.
    """

    data = request.get_json()
    transcript = data.get("transcript")
    summary_type = data.get("summary_type", "Detailed")
    language = data.get("language", "English")
    style = data.get("style", "Formal")

    if not transcript:
        return jsonify({"error": "Transcript is required"}), 400

    summary_template = "Detailed summary with key points." if summary_type.lower() == "detailed" else "Concise summary in 100 words."

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

    if result == "Error generating summary.":
        return jsonify({"error": "Summary generation failed. Please try again later."}), 500

    return jsonify({"summary": result}), 200

# Route for generating tags from transcript
@app.route('/generate_tags', methods=['POST'])
def generate_tags():
    """
    Extract relevant tags (keywords) from a transcript.
    """

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
    try:
        result = chain.invoke({"transcript": transcript})
    except Exception as e:
        print(f"Tag generation failed: {e}")
        return jsonify({"error": f"Tag generation failed: {str(e)}"}), 500

    # Step 5: Return clean JSON
    return jsonify(result), 200

# Route for extracting frames and analyzing them
@app.route('/frames_description', methods=['POST'])
def frames_analysis():
    """
    Analyze and describe frames from a video.
    """
    data = request.get_json()
    video_path = data.get("video_path")
    language = data.get("language", "English") 

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Invalid video path"}), 400

    # Step 1: Extract and encode frames
    frames = get_frames(video_path)

    if not frames:
        return jsonify({"error": "Failed to extract frames from video."}), 500

    # Step 2: Analyze frames using OpenAI tool-calling function
    analysis_results = analysis_with_openAI_vanilla(frames, language)

    # Step 3: Add base64 back into each frame for frontend display
    for i, result in enumerate(analysis_results):
        result["image"] = frames[i]["image"]  # Add base64-encoded image to each result
    
    # Step 4: Return in wrapped JSON format
    return jsonify({
        "frames": analysis_results
    })

# Route for holistic summary generation
@app.route("/holistic_summary", methods=["POST"])
def holistic_summary():
    """
    Generate a holistic summary that combines transcript and visual analysis.
    """
    data = request.get_json()
    transcript = data.get("transcript", "")
    frames = data.get("frames", [])
    language = data.get("language", "English")

    if not transcript or not frames:
        return jsonify({"error": "Transcript and frame descriptions are required"}), 400

    # Combine frame descriptions
    frame_descriptions = "\n".join([
        f"{frame.get('image_name', 'Frame')}:\n{frame.get('description', '')}"
        for frame in frames
    ])

    # Compose the full content to summarize
    full_content = f"Transcript:\n{transcript}\n\nFrame Descriptions:\n{frame_descriptions}"

    # System prompt template
    system_message = f'''
        You are an expert media analyst. Given a full transcript and detailed visual descriptions from a video, 
        write a comprehensive and holistic summary that combines both elements. Write the summary in {language}.

        Follow these steps in your analysis:

        1. **Identify the Main Topic**: Start by identifying the central topic or theme of the video.
        2. **Classify the Video Type**: Determine the type of video (e.g., tutorial, documentary, interview, entertainment, etc.), based on both the transcript and visual content.
        3. **Transcript Summary**: Provide a concise summary of the transcript, capturing the key points and the overall message of the spoken content.
        4. **Frame Analysis**: Summarize the key frames from the video, highlighting the most significant visual moments, and describe their relevance to the content of the video.
        5. **Final Holistic Summary**: Integrate the topic, video type, transcript summary, and frame analysis into a unified, informative summary that reflects both the visual and verbal content. The summary should be concise yet comprehensive, capturing the essence of the video in a clear and informative manner.

        Keep the summary concise but informative, and ensure that it covers all the important aspects of the video content.
    '''

    # Prepare prompt using ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{content}")
    ])

    try:
        # Format the message and invoke the model
        prompt = prompt_template.format_messages(content=full_content)
        response = model.invoke(prompt)
        # Get the result and return it
        result = response.content
    except Exception as e:
        print(f"Holistic summary generation failed: {e}")
        return jsonify({"error": f"Holistic summary generation failed: {str(e)}"}), 500

    return jsonify({"holistic_summary": result}), 200

if __name__ == '__main__':
    app.run(debug=True)