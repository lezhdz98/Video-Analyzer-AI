from flask import Flask, jsonify
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key is not set in the environment variables.")

# Initialize the chat model with the API key
model = init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=api_key)

# Optional: print the API key to verify it's loaded correctly
print(f"OpenAI API Key: {api_key}")

# Define system template for video analysis
system_template = "Analyze the following video transcript and summarize the key points."

# Define user template for transcript
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{transcript}")]
)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Hello from Flask backend!"})

if __name__ == '__main__':
    app.run(debug=True)