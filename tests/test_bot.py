from src.contextual_brain import ContextualKnowledgeBase

csv_path = r"C:\Prog\CHATBOT\data\wikiqa\wikiqa_final.csv"
index_path = r"C:\Prog\CHATBOT\embeddings\wikiqa_faiss.index"

kb = ContextualKnowledgeBase(csv_path, index_path)

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    print("Bot:", kb.query(query))
