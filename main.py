from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai
from routes import upload, predict
from database.connection import engine
from models.scan import Base

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

app.include_router(upload.router)
app.include_router(predict.router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chatbot/", response_model=ChatResponse)
async def chatbot_endpoint(request: ChatRequest):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(request.message)

        if hasattr(response, "text"):
            return ChatResponse(response=response.text.strip())
        elif hasattr(response, "candidates") and response.candidates:
            return ChatResponse(response=response.candidates[0].text.strip())
        else:
            return ChatResponse(response="I'm not sure how to respond to that.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")