from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add the project root to the system path so we can import the AI module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.companion import get_ai_response

# Initialize the FastAPI app
app = FastAPI(title="MindCompanion API")

# Define the format for incoming messages
class ChatRequest(BaseModel):
    message: str

# Create an endpoint to talk to the AI
@app.post("/chat")
async def chat_with_companion(request: ChatRequest):
    # Pass the user's message to the AI and get the reply
    reply = get_ai_response(request.message)
    return {"reply": reply}

@app.get("/")
async def root():
    return {"message": "MindCompanion API is running!"}