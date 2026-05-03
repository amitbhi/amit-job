"""
Optimized RAG Engine for Amit Bhise Recruiter Chatbot
Uses Direct Context Injection for maximum reliability on free-tier hosting (Render).
Eliminates memory-heavy local embedding models.
"""

import os
from typing import Dict
import groq


class RAGEngine:
    def __init__(self, groq_api_key: str, knowledge_base_path: str = "knowledge_base.txt"):
        """Initialize RAG engine with direct knowledge base loading"""
        
        # Initialize Groq client
        self.groq_client = groq.Groq(api_key=groq_api_key)
        self.knowledge_base_path = knowledge_base_path
        
        # Load knowledge base once into memory
        self.knowledge_content = self._load_full_knowledge()
        print(f"RAG Engine: Knowledge base loaded ({len(self.knowledge_content)} characters)")
    
    def _load_full_knowledge(self) -> str:
        """Load the entire knowledge base file"""
        if not os.path.exists(self.knowledge_base_path):
            return "Knowledge base not found."
            
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def generate_answer(self, query: str) -> str:
        """Generate answer using Groq API with full context injection"""
        
        # System prompt with core instructions
        system_prompt = """You are an AI talent agent representing Amit Bhise (AB), an Artificial Intelligence Leader.

Your goal is to position Amit as a high-impact, enterprise-grade AI leader suitable for senior roles (Director, VP, Head of AI, Chief AI Officer).

Follow these guidelines in all your responses:

Strategic Positioning:
- Highlight measurable business outcomes (ROI, cost savings, efficiency gains)
- Emphasize leadership, scale, and global experience
- Showcase expertise in GenAI, Agentic AI, and enterprise AI platforms
- Tailor answers to recruiter intent—identify why they are asking and address the underlying business need

Communication Standards:
- Be concise, confident, and business-focused
- Use an executive-level, outcome-driven tone
- For contact information, provide email: amitvishnu@gmail.com and phone: +91 98908 97285

Avoid:
- Generic or academic responses
- Overly technical explanations unless explicitly asked
- Underselling impact or focusing on experimentation over results"""

        user_prompt = f"""Use the following comprehensive professional profile to answer the recruiter's question.

AMIT BHISE PROFESSIONAL PROFILE:
{self.knowledge_content}

RECRUITER QUESTION: {query}

Please provide a helpful, accurate, and strategically positioned answer based on the profile above."""

        # Call Groq API
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=1024,
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def query(self, question: str) -> Dict[str, any]:
        """Main query method - compatible with previous API"""
        answer = self.generate_answer(question)
        
        return {
            "question": question,
            "answer": answer,
            "sources": 1  # Full context used as single source
        }

    @property
    def collection(self):
        """Mock collection property for API compatibility"""
        class MockCollection:
            def count(self):
                return 1
        return MockCollection()
