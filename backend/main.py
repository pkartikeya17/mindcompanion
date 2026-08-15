from fastapi import FastAPI

# Initialize the backend application
app = FastAPI(title="MindCompanion API")

# Create a simple route to verify the server is working
@app.get("/")
def home():
    return {"message": "Hello, MindCompanion is running!"}