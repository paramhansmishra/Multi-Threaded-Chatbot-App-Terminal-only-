def retrieve_with_scores(self, query, top_k=3):
    query_vec = self.embedder.encode([query])
    D, I = self.index.search(query_vec, top_k)
    return [(self.questions[i], D[0][j]) for j, i in enumerate(I[0])]
