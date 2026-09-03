"""
RAG pipeline: FAISS retrieval (built in kaggle_notebooks/04_rag_build_index.ipynb)
+ Groq LLM generation.
"""
import json
import os
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from groq import Groq

MODELS_DIR = Path(__file__).parent.parent / "models" / "rag"

SYSTEM_PROMPT_TEMPLATE = """You are a helpful, professional customer support assistant for an online retailer. Answer the customer's question using ONLY the information in the retrieved support responses below. If the customer sounds frustrated ({detected_sentiment}), acknowledge that before answering. If the retrieved context does not cover the question, say so honestly and offer to escalate to a human agent rather than guessing."""


class RAGPipeline:
    def __init__(self, groq_api_key: str, groq_model: str = "openai/gpt-oss-120b"):
        self.index = faiss.read_index(str(MODELS_DIR / "faiss_index.bin"))

        with open(MODELS_DIR / "chunks.json") as f:
            self.chunks = json.load(f)

        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_model = groq_model

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "score": float(score),
                "instruction": chunk["instruction"],
                "response": chunk["response"],
                "intent": chunk["intent"],
                "category": chunk["category"],
            })
        return results

    def generate(self, user_message: str, sentiment_label: str, k: int = 3) -> dict:
        retrieved = self.retrieve(user_message, k=k)

        context_block = "\n\n".join(
            f"[{i+1}] {r['response']}" for i, r in enumerate(retrieved)
        )

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(detected_sentiment=sentiment_label)

        user_prompt = f"""Context (retrieved past support responses):
{context_block}

Customer question: "{user_message}\""""

        completion = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )

        answer = completion.choices[0].message.content

        return {
            "answer": answer,
            "retrieved_chunks": retrieved,
        }
