"""
Main FastAPI app for the RAG e-commerce support chatbot.

Run locally with:
    uvicorn app.main:app --reload --port 8000

Requires model artifacts downloaded from the Kaggle notebooks and placed under:
    local_app/models/language/
    local_app/models/sentiment/
    local_app/models/intent/
    local_app/models/rag/

And a .env file (copy from .env.example) with your OWN rotated Groq API key.
"""
import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.language import LanguageDetector
from app.sentiment import SentimentClassifier, MODELS_DIR as SENTIMENT_DIR
from app.intent import IntentClassifier
from app.rag import RAGPipeline
from app.router import route_message, APOLOGY_PREFIX

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

# Global model holders, populated at startup
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")

    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and fill in your "
            "OWN rotated Groq API key."
        )

    models["language"] = LanguageDetector()
    logger.info("Language detector loaded.")

    models["sentiment"] = None
    if (SENTIMENT_DIR / "bucket_labels.json").exists() and (
        SENTIMENT_DIR / "config.json"
    ).exists():
        models["sentiment"] = SentimentClassifier()
        logger.info("Sentiment classifier loaded.")
    else:
        logger.warning("Sentiment model not found — skipping. Using neutral fallback.")

    models["intent"] = IntentClassifier()
    logger.info("Intent classifier loaded.")

    models["rag"] = RAGPipeline(groq_api_key=groq_api_key, groq_model=groq_model)
    logger.info("RAG pipeline loaded.")

    logger.info("All models loaded. Ready.")
    yield
    models.clear()


app = FastAPI(title="E-commerce Support Chatbot", lifespan=lifespan)

# Open CORS for local Next.js frontend development.
# Tighten this (specific origins) before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    detected_language: str
    sentiment: str
    sentiment_confidence: float
    intent: str
    intent_route: str
    intent_confidence: float
    priority_flag: bool
    handling: str
    retrieved_chunks: list[dict] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    text = req.message.strip()

    # Stage 1: language
    language = models["language"].predict(text)

    # Stage 2: sentiment
    if models["sentiment"] is not None:
        sentiment_result = models["sentiment"].predict(text)
    else:
        sentiment_result = {"label": "neutral", "confidence": 0.0}

    # Stage 3: intent
    intent_result = models["intent"].predict(text)

    # Stage 4: route
    routing = route_message(intent_result, sentiment_result)

    retrieved_chunks = None

    if routing["handling"] == "rag":
        rag_result = models["rag"].generate(
            user_message=text,
            sentiment_label=sentiment_result["label"],
        )
        answer = rag_result["answer"]
        retrieved_chunks = rag_result["retrieved_chunks"]

        if routing.get("needs_apology_prefix"):
            answer = APOLOGY_PREFIX + answer

        final_response = answer
    else:
        final_response = routing["response"]

    return ChatResponse(
        response=final_response,
        detected_language=language,
        sentiment=sentiment_result["label"],
        sentiment_confidence=sentiment_result["confidence"],
        intent=intent_result["fine_intent"],
        intent_route=intent_result["route"],
        intent_confidence=intent_result["confidence"],
        priority_flag=routing["priority_flag"],
        handling=routing["handling"],
        retrieved_chunks=retrieved_chunks,
    )
