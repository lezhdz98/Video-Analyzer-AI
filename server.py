from flask import Flask, jsonify
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API Key: {api_key}")  # Print API key to terminal

@app.route('/')
def home():
    return jsonify({"message": "Hello from Flask backend!"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)