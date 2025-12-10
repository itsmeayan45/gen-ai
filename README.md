# Gen AI Projects

A collection of AI-powered chatbot applications built with LangChain, Streamlit, and OpenRouter API.

## 🚀 Projects

### 1. Q&A Chatbot

**Location:** `Q&A_chatbot/`

A simple question-answering chatbot with a clean Streamlit interface.

**Features:**

- Direct conversational interface
- Powered by free OpenRouter models (Google Gemini Flash, Mistral)
- Clean and intuitive UI
- Configurable response parameters

**Files:**

- `app.py` - Main Streamlit application
- `test_api.py` - API connection testing
- `find_free_models.py` - Discover available free models
- `ollamaapp.py` - Ollama integration variant

**Run:**

```bash
cd Q&A_chatbot
streamlit run app.py
```

---

### 2. RAG Q&A Chatbot

**Location:** `rag_q&a_chatbot/`

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from uploaded PDF documents.

**Features:**

- PDF document upload and processing
- FAISS vector store for efficient retrieval
- Custom embeddings implementation
- Context-aware responses from documents
- Question answering with source context

**Files:**

- `ragapp.py` - Main RAG application
- `pdfs/` - Directory for PDF documents

**Run:**

```bash
cd rag_q&a_chatbot
streamlit run ragapp.py
```

---

### 3. Conversational Q&A Chatbot with PDF

**Location:** `conversational_q&a_chatbot/`

An advanced conversational chatbot with PDF support and persistent chat history.

**Features:**

- Upload and query PDF documents
- Session-based chat history
- Conversational memory (remembers previous messages)
- RAG-based responses with context
- Streamlit Cloud ready
- Free AI models via OpenRouter

**Files:**

- `app.py` - Main application
- `requirements.txt` - Python dependencies
- `README.md` - Project-specific documentation
- `DEPLOYMENT.md` - Streamlit Cloud deployment guide
- `.streamlit/config.toml` - Streamlit configuration
- `.gitignore` - Git exclusions

**Run:**

```bash
cd conversational_q&a_chatbot
streamlit run app.py
```

**Deploy to Streamlit Cloud:**
See `conversational_q&a_chatbot/DEPLOYMENT.md` for detailed deployment instructions.

---

## 🛠️ Tech Stack

- **Python 3.13.7**
- **Streamlit** - Web UI framework
- **LangChain** - LLM framework and chains
- **OpenRouter** - AI model API with free tier
- **FAISS** - Vector database for embeddings
- **PyPDF** - PDF document processing
- **NumPy** - Numerical operations for embeddings
- **python-dotenv** - Environment variable management

## 📋 Prerequisites

- Python 3.13 or higher
- OpenRouter API key (get it free at [openrouter.ai](https://openrouter.ai/))
- Virtual environment (recommended)

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/itsmeayan45/gen-ai.git
cd gen-ai
```

### 2. Create Virtual Environment

```bash
python -m venv genvenv
genvenv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here  # Optional
```

### 5. Run Any Project

Navigate to the project folder and run:

```bash
streamlit run app.py
```

Or for the RAG chatbot:

```bash
streamlit run ragapp.py
```

## 🌐 Free AI Models

All projects use free OpenRouter models:

- `google/gemini-2.0-flash-exp:free`
- `mistralai/mistral-7b-instruct:free`

No API costs required for basic usage!

## 📁 Project Structure

```
Gen-ai-projects/
├── Q&A_chatbot/              # Basic Q&A chatbot
│   ├── app.py
│   ├── test_api.py
│   └── find_free_models.py
├── rag_q&a_chatbot/          # RAG chatbot with PDF
│   ├── ragapp.py
│   └── pdfs/
├── conversational_q&a_chatbot/  # Advanced conversational chatbot
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   ├── DEPLOYMENT.md
│   └── .streamlit/
├── genvenv/                  # Virtual environment
├── .env                      # API keys (not in git)
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## 🚢 Deployment

The **Conversational Q&A Chatbot** is ready for deployment on Streamlit Cloud:

1. Push to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io/)
3. Add `OPENROUTER_API_KEY` in Streamlit secrets
4. Deploy!

See `conversational_q&a_chatbot/DEPLOYMENT.md` for detailed steps.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ayan**

- GitHub: [@itsmeayan45](https://github.com/itsmeayan45)

## 🙏 Acknowledgments

- LangChain for the amazing framework
- OpenRouter for free AI model access
- Streamlit for the easy-to-use web framework

---

**Star ⭐ this repository if you find it helpful!**
