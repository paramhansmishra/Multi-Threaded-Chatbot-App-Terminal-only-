import socket
import threading
from src.contextual_brain import ContextualKnowledgeBase

# Load the retrieval+LLM brain once when the server starts
CSV_PATH = r"C:\Prog\CHATBOT\data\wikiqa\wikiqa_final.csv"
INDEX_PATH = r"C:\Prog\CHATBOT\embeddings\wikiqa_faiss.index"
brain = ContextualKnowledgeBase(CSV_PATH, INDEX_PATH)

# Basic config
HOST = "127.0.0.1"
PORT = 5000

def handle_client(conn, addr):
    print(f"🧩 Connected with {addr}")
    conn.sendall(b"Welcome to the AI Bot Server! Type 'exit' to quit.\n")

    try:
        while True:
            data = conn.recv(1024).decode().strip()
            if not data or data.lower() == "exit":
                break

            print(f"[{addr}] User: {data}")
            response = brain.query(data)
            conn.sendall((response + "\n").encode())

    except Exception as e:
        print(f"⚠️  Error handling {addr}: {e}")
    finally:
        conn.close()
        print(f"🔌 Disconnected {addr}")

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen()
        print(f"🚀 Server running on {HOST}:{PORT}")
        while True:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.start()
            print(f"👥 Active connections: {threading.active_count() - 1}")

if __name__ == "__main__":
    start_server()
