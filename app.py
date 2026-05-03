"""
FastAPI Backend for Amit Bhise Recruiter Chatbot
Provides REST API endpoints for the frontend to query
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from rag_engine import RAGEngine

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Amit Bhise Recruiter Chatbot API",
    description="RAG-powered chatbot to answer questions about Amit Bhise's professional profile",
    version="1.0.0"
)

# Enable CORS for frontend (HTML/JS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is Amit's experience with AI and machine learning?"
            }
        }

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: int

# Initialize RAG engine
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set!")

rag_engine = RAGEngine(groq_api_key=GROQ_API_KEY)

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Amit Bhise Recruiter Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Submit a question about Amit Bhise",
            "GET /health": "Check API health status",
            "GET /docs": "API documentation (Swagger UI)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_engine": "initialized",
        "vector_db_chunks": 1 # Simplified for direct injection
    }

@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """
    Query the RAG chatbot about Amit Bhise
    
    Args:
        request: QueryRequest with question field
    
    Returns:
        QueryResponse with answer and metadata
    """
    try:
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Query RAG engine
        result = rag_engine.query(request.question)
        
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/sample-questions")
async def sample_questions():
    """Get sample questions recruiters can ask"""
    return {
        "sample_questions": [
            "What is Amit's experience with AI and machine learning?",
            "Tell me about Amit's key achievements at Suzlon Energy",
            "What technical skills does Amit have?",
            "What is Amit's educational background?",
            "Has Amit published any research papers?",
            "What programming languages and tools does Amit know?",
            "What is Amit's experience with MLOps?",
            "Tell me about Amit's leadership experience",
            "What industries has Amit worked in?",
            "How can I contact Amit?",
            "Is Amit open to relocation?",
            "What are Amit's interests outside of work?"
        ]
    }

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
