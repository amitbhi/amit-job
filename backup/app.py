from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import torch

app = Flask(__name__)
CORS(app)

# ---- Configuration ----
MODEL_NAME = "MBZUAI/LaMini-Flan-T5-77M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Simple Session Memory (Global for local dev)
chat_history = []

try:
    print(f"Loading models on {DEVICE}...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    
    def load_documents():
        if not os.path.exists('context.txt'):
            return ["No context available."]
        with open('context.txt', 'r') as f:
            content = f.read()
        return [p.strip() for p in content.split('\n\n') if p.strip()]

    documents = load_documents()
    doc_embeddings = embedder.encode(documents)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(np.array(doc_embeddings).astype('float32'))
    print("Memory and Neural Core ready.")

except Exception as e:
    print(f"STARTUP ERROR: {e}")
    exit(1)

def get_ai_response(query, context, history_context=""):
    # Ultra-clear identity prompt
    prompt = f"""You are a helpful AI Assistant. Your only job is to talk about the person "Amit Bhise".
AMIT BHISE IS A PERSON, A LEADER, AND A RESEARCHER. YOU ARE THE AI BOT. 

Context about Amit Bhise:
{context}

Recent conversation:
{history_context}

User: {query}
AI Assistant Response (Talk about Amit in the third person):"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.3, do_sample=True, repetition_penalty=1.5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history
    data = request.json
    query = data.get('query', '').strip()
    query_lower = query.lower()
    
    # --- Layer 1: Memory-Aware Greeting Router ---
    greetings = ['hi', 'hello', 'hey', 'morning', 'evening']
    affirmations = ['yes', 'sure', 'ok', 'okay', 'yep', 'tell me more']
    
    # Handle simple greetings
    if any(g == query_lower for g in greetings):
        response = "Hello! I'm Amit's AI Assistant. I'd be delighted to introduce you to him. Amit is a distinguished AI leader in Pune with 20 years of experience, but he's also a family man from Phaltan who is passionate about philosophy and Bertolt Brecht theatre. Would you like to hear about his professional milestones at Suzlon and TCS, or more about his personal interests?"
        chat_history = [f"User: {query}", f"Assistant: {response}"]
        return jsonify({"response": response})

    # Handle "Yes/Sure" by looking at history
    if any(a == query_lower for a in affirmations) and chat_history:
        # If user says yes to the greeting offer
        context = "\n".join(documents[:5]) # Provide a broad career overview context
        response = get_ai_response("Tell me about Amit's professional career and achievements.", context)
        chat_history.append(f"User: {query}")
        chat_history.append(f"Assistant: {response}")
        return jsonify({"response": response})

    # --- Layer 2: RAG for Factual Queries ---
    try:
        query_embedding = embedder.encode([query])
        distances, indices = index.search(np.array(query_embedding).astype('float32'), 4)
        retrieved_docs = [documents[i] for i in indices[0]]
        context = "\n".join(retrieved_docs)
        
        history_context = "\n".join(chat_history[-2:]) # Keep last turn for context
        answer = get_ai_response(query, context, history_context)
        
        # Update history
        chat_history.append(f"User: {query}")
        chat_history.append(f"Assistant: {answer}")
        if len(chat_history) > 6: chat_history = chat_history[-6:]
        
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
