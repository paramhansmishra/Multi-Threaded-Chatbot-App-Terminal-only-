# 🧠 Multi-Threaded AI Chatbot Server

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FAISS](https://img.shields.io/badge/FAISS-Enabled-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

A **multi-threaded conversational AI system** that allows multiple clients to chat concurrently with an intelligent server.  
Each client interacts in real time with a **Contextual Knowledge Base (FAISS-powered retriever)** that falls back to a **Large Language Model (LLM)** when no good match is found.

---

## ⚙️ Features

- Multi-threaded socket server — handles multiple clients concurrently  
- Real-time message exchange over TCP  
- Semantic QA retrieval with **FAISS + SentenceTransformer**  
- **LLM fallback** via OpenRouter (Mistral 7B, Gemini, etc.)  
- Persistent embeddings for fast startup  
- Modular, extensible architecture

---

## 🧩 System Overview
### Architecture
      ┌──────────────────────────────────────────┐
      │              Server (AI Brain)           │
      │------------------------------------------│
      │ ContextualKnowledgeBase (FAISS + LLM)    │
      │ Thread 1 ─ handle_client(conn_1)         │
      │ Thread 2 ─ handle_client(conn_2)         │
      │ Thread n ─ handle_client(conn_n)         │
      └──────────────────────────────────────────┘
               ↑                    ↑
               │                    │
    ┌──────────┴───────────┐  ┌─────┴─────────┐
    │     Client #1        │  │   Client #2   │
    │ (Socket + Input Loop)│  │ (Socket Loop) │
    └──────────────────────┘  └───────────────┘

---

## 🧱 Core Components

### `multi_threaded_server.py`
- Initializes TCP socket (`127.0.0.1:5000`)
- Loads **ContextualKnowledgeBase** (FAISS index)
- Spawns new thread for every incoming client  
- Handles message receive, query, and send cycle  
- Each client is isolated and concurrent

### `client.py`
- Connects to server and starts a user input loop  
- Sends text queries and receives responses  
- Supports `"exit"` command to disconnect gracefully

### `contextual_brain.py`
- Loads the WikiQA dataset (`wikiqa_final.csv`)  
- Uses **SentenceTransformer ("multi-qa-MiniLM-L6-cos-v1")** to embed questions  
- Builds or loads a **FAISS inner product index** for fast similarity search  
- Returns best answers above threshold; otherwise calls **LLM fallback**

### `llm_fallback.py`
- Calls OpenRouter API for Mistral-7B-Instruct responses  
- If no API key or network, gracefully degrades to local text stub  

### `test_bot.py`
- CLI test harness for querying the ContextualKnowledgeBase directly  
- Useful for validating retrieval + LLM logic before server integration  

---

## 🧠 Query Flow

1. Client sends message → Server receives via socket  
2. Server passes message to `ContextualKnowledgeBase.query()`  
3. System embeds the query and retrieves top matches from FAISS  
4. If similarity score ≥ 0.6 → returns best answers  
5. Else → calls `get_llm_response()` (LLM fallback)  
6. Server sends response back to client  

---

## 🧾 Example Interaction

**Client**

    Connected to server at 127.0.0.1:5000
    You: What is machine learning?
    Bot: (Top score: 0.845)

    (0.845) Machine learning is a subset of AI where systems learn patterns from data.
    You: Who founded OpenAI?
    Bot: OpenAI was founded by Elon Musk, Sam Altman, and others in 2015.
    You: exit
  
**Server Console**

    🚀 Server running on 127.0.0.1:5000
    📚 Loading dataset: data/wikiqa_final.csv
    📦 Loading existing FAISS index...
    ✅ ContextualKnowledgeBase ready with 1040 entries.
    🧩 Connected with ('127.0.0.1', 53421)
    [('127.0.0.1', 53421)] User: What is machine learning?
    👥 Active connections: 1
    🔌 Disconnected ('127.0.0.1', 53421)

---

## 📦 Repository Structure

    CHATBOT
    │   faiss_test.py
    │   getwikiqa.py
    │   main.py
    │   README.md
    │   requirements.txt
    │   __init__.py
    │
    ├───data
    │   │   faq_data.csv
    │   │
    │   └───wikiqa
    │       │   wikiqa.csv
    │       │   wikiqa_clean.csv
    │       │   wikiqa_final.csv
    │       │
    │       ├───processed
    │       │       prepare_wikiqa.py
    │       │
    │       └───raw
    ├───embeddings
    │       faiss_index.bin
    │       questions.npy
    │       wikiqa_faiss.index
    │
    ├───src
    │   │   bot_brain.py
    │   │   contextual_brain.py
    │   │   embedder.py
    │   │   llm_fallback.py
    │   │   multi_threaded_server.py
    │   │   retriever.py
    │   │   __init__.py
    │   │
    │   └───__pycache__
    │           bot_brain.cpython-313.pyc
    │           contextual_brain.cpython-313.pyc
    │           llm_fallback.cpython-313.pyc
    │           multi_threaded_server.cpython-313.pyc
    │           __init__.cpython-313.pyc
    │
    ├───tests
    │   │   client.py
    │   │   test_bot.py
    │   │   test_faiss.py
    │   │   __init__.py
    │   │
    │   └───__pycache__
    │           test_bot.cpython-313.pyc
    │           __init__.cpython-313.pyc
    │
    └───__pycache__
            __init__.cpython-313.pyc
  
---

## 🧠 Setup & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/multi-threaded-chatbot.git
cd multi-threaded-chatbot
pip install sentence-transformers faiss-cpu pandas numpy requests scikit-learn
python multi_threaded_server.py
python client.py
```
You can start multiple clients simultaneously — each runs in its own thread on the server.

---
🔐 Environment Variable (Optional)

To use live LLM fallback, set your OpenRouter API key:
```bash
setx OPENROUTER_API_KEY "sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
---

🧮 Future Enhancements

| Area                     | Planned Upgrade                                    |
| ------------------------ | -------------------------------------------------- |
| **Async Architecture**   | Replace threads with `asyncio` for scalable I/O    |
| **User Authentication**  | Add unique user IDs and session history            |
| **Logging & Analytics**  | Store conversation logs for NLP analytics          |
| **WebSocket Layer**      | Build browser-accessible version with `websockets` |
| **GUI Client**           | Add Tkinter or React-based chat interface          |
| **Fine-tuned QA Models** | Integrate custom domain QA models                  |

---

👤 Author

    Developer: Paramhans Mishra

    Email: param110045@gmail.com

    GitHub Portfolio: github.com/paramhansmishra

Explore my GitHub for in-depth AI, ML, and system design projects — including CNN-based ingredient analysis, face recognition, and retrieval-augmented bots.
---
