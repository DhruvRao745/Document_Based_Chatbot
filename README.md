# Document Based Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that lets you upload any PDF and ask questions about it in natural language. Built with LangChain, FAISS, Groq LLaMA 3, and Streamlit.

---

## What it does

- Upload any PDF document (resume, research paper, report, etc.)
- Ask questions in natural language
- Get accurate answers grounded strictly in the document content
- Maintains full chat history within the session

---

## How it works

```
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Vector Store
                                                              ↓
User Question → Embed Query → Similarity Search → Top-k Chunks → LLaMA 3 → Answer
```

1. **Ingestion** — PDF text is extracted using `pdfplumber`, split into overlapping chunks, and embedded using `sentence-transformers/all-MiniLM-L6-v2`
2. **Storage** — Embeddings are stored in a FAISS vector index for fast similarity search
3. **Retrieval** — On each question, the query is embedded and the top-k most relevant chunks are retrieved
4. **Generation** — Retrieved chunks are passed as context to LLaMA 3 via Groq API, which generates a grounded answer

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Extraction | pdfplumber |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| LLM | LLaMA 3.3 70B via Groq API |
| RAG Framework | LangChain |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/DhruvRao745/Document_Based_Chatbot.git
cd Document_Based_Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) → Create API Key (free, no credit card)

### 5. Create a `.env` file

```
GROQ_API_KEY=your_groq_api_key_here
```

### 6. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. Upload a PDF using the sidebar
2. Wait for the "Indexed X chunks!" confirmation
3. Type your question in the chat box
4. The bot answers using only the content from your document

---

## Project Structure

```
Document_Based_Chatbot/
├── app.py              # Streamlit UI and session management
├── rag_pipeline.py     # PDF loading, chunking, embedding, RAG chain
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md
```

---

## Key Concepts Demonstrated

- **RAG architecture** — combining retrieval with generation for accurate, grounded answers
- **Vector embeddings** — converting text to semantic vectors using transformer models
- **FAISS indexing** — efficient similarity search over large document collections
- **LangChain** — orchestrating the full pipeline from document ingestion to LLM response
- **Streamlit** — building and deploying interactive ML applications

---

## Future Improvements

- [ ] Support for multiple PDF uploads
- [ ] Add source chunk highlighting to show where answers came from
- [ ] Persistent vector store (save index to disk)
- [ ] Deploy to Streamlit Cloud

---

## Author

**Dhruv Kumar Rao**  
[GitHub](https://github.com/DhruvRao745) • Email: dhruvrao434@gmail.com
