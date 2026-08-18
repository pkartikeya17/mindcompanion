import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. Import our newly built RAG Retriever
from ai.rag.retriever import get_relevant_context

# 2. Load the vault
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("AI_API_KEY")

# 3. Connect using the Gemini SDK
client = genai.Client(api_key=api_key)

# 4. Safety guardrails & companion persona (Updated for RAG)
safety_instructions = """
You are a compassionate, empathetic, and supportive mental health companion. 
Your goal is to listen, validate feelings, and provide emotional support.

CRITICAL RULES:
1. You are NOT a doctor, therapist, or medical professional.
2. NEVER diagnose a condition, prescribe treatments, or give medical advice.
3. If someone is in crisis, gently but firmly encourage them to seek real professional help or contact emergency services.
4. Keep your responses concise, warm, and conversational.
5. EVIDENCE-BASED GROUNDING: You will be provided with clinical context from evidence-based guidelines (WHO). Use this context to inform your supportive advice. If the context contains helpful exercises (like grounding or breathing), gently suggest them. Do not hallucinate clinical information outside of this context.
"""

def get_ai_response(user_message: str) -> str:
    try:
        # Step A: Retrieve relevant knowledge from FAISS based on the user's message
        print(f"Searching Vector DB for context regarding: '{user_message}'")
        retrieved_context = get_relevant_context(user_message, top_k=3)
        
        # Step B: Combine the context and the user message into a single prompt
        prompt = f"""
        Here is some relevant clinical context from our knowledge base:
        {retrieved_context}

        User's message:
        "{user_message}"
        
        Please respond to the user following your safety instructions.
        """

        # Step C: Send the combined prompt to Gemini
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=safety_instructions,
            )
        )
        return response.text
        
    except Exception as e:
        # This will print the exact error to your VS Code terminal
        print(f"Error calling Gemini API or Retriever: {e}")
        # Return a graceful fallback message so FastAPI doesn't crash
        return "I am currently experiencing technical difficulties. Please try again later."