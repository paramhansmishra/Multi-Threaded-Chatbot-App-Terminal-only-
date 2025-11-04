import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.llm_fallback import get_llm_response

class ContextualKnowledgeBase:
    def __init__(self, csv_path: str, index_path: str = "embeddings/wikiqa_faiss.index"):
        print(f"📚 Loading dataset: {csv_path}")
        self.data = pd.read_csv(csv_path)

        if 'label' in self.data.columns:
            self.data = self.data[self.data['label'] == 1]

        self.data = self.data.dropna(subset=['question', 'answer']).drop_duplicates()
        self.questions = self.data['question'].tolist()
        self.answers = self.data['answer'].tolist()

        # Load SentenceTransformer model fine-tuned for QA retrieval
        self.model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

        self.index_path = index_path
        if os.path.exists(index_path):
            print("📦 Loading existing FAISS index...")
            self.index = faiss.read_index(index_path)
        else:
            print("⚙️  Creating new FAISS index...")
            self.build_faiss_index()

        print(f"✅ ContextualKnowledgeBase ready with {len(self.data)} entries.")

    def build_faiss_index(self):
        """Encode questions and store vectors in FAISS index."""
        embeddings = self.model.encode(self.questions, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product on normalized vectors
        self.index.add(embeddings)

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        print(f"💾 Saved FAISS index to {self.index_path}")

    def query(self, user_input: str, top_k: int = 2) -> str:
        """Retrieve top semantic matches from FAISS index."""
        user_vec = self.model.encode([user_input], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(user_vec, top_k)

        scores = scores.flatten()
        indices = indices.flatten()

        results = []
        for idx, score in zip(indices, scores):
            if 0 <= idx < len(self.answers):
                ans = self.answers[idx].strip()
                if ans not in results:  # prevent duplicates
                    results.append((score, ans))

        best_score = scores[0] if len(scores) else 0.0

        if best_score > 0.6:
            answer_text = "\n".join([f"- ({s:.3f}) {a}" for s, a in results])
            return f"(Top score: {best_score:.3f})\n{answer_text}"
        else:
            print(f"Best score: {best_score:.3f} .No good match found in knowledge base, falling back to LLM.")
            return get_llm_response(user_input)