import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBase:
    def __init__(self, csv_path: str):
        print(f"📚 Loading data from: {csv_path}")

        # Load dataset
        self.data = pd.read_csv(csv_path)

        # Filter only correct answers if 'label' column exists
        if 'label' in self.data.columns:
            self.data = self.data[self.data['label'] == 1]

        # Drop NaN or duplicates to clean data
        self.data = self.data.dropna(subset=['question', 'answer']).drop_duplicates()

        # Limit rows for performance (optional)
        self.data = self.data.head(3000)

        # Build TF-IDF model
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.question_vectors = self.vectorizer.fit_transform(self.data['question'])

        print(f"✅ KnowledgeBase initialized with {len(self.data)} question-answer pairs.")

    def query(self, user_input: str) -> str:
        """Return one or more relevant answers based on similarity score."""
        input_vec = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(input_vec, self.question_vectors)[0]

        # Find best score and its close neighbors
        best_score = similarities.max()
        close_matches = (similarities >= best_score - 0.02)  # 0.02 = tolerance margin

        results = self.data[close_matches]
        answers = results['answer'].unique()[:3]  # return up to 3 to avoid flooding

        if best_score > 0.35:
            joined = "\n".join(
                [f"- {ans.strip()}" for ans in answers if isinstance(ans, str)]
            )
            return f"(Top score: {best_score:.3f})\n{joined}"
        else:
            return f"(Score: {best_score:.3f}) I'm not sure about that — let me look it up."
