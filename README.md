---

# 🏨 Hotel-Recommendation-ChatBot-for-India

A smart hotel recommendation chatbot tailored for India, powered by LLMs, RAG (Retrieval-Augmented Generation), and vector databases. It provides personalized hotel suggestions based on real hotel data, enabling natural and helpful interactions with users.

---

## 📊 Dataset

The core knowledge base is built from the [Indian Hotels on Goibibo](https://www.kaggle.com/datasets/PromptCloudHQ/hotels-on-goibibo) dataset. This includes vital information such as hotel names, locations, descriptions, and other metadata crucial for recommendations.

### 🔧 Data Preparation

* Raw CSV data is cleaned and pre-processed.
* Each hotel entry is transformed into a natural language document.
* These documents are embedded and stored in **ChromaDB** for fast retrieval using semantic similarity.

---

## ⚙️ Implementation Overview

Built with the **LangChain** framework and designed for modularity, the system includes:

### 🔹 1. **Large Language Models (LLMs)**

Compatible with Openrouter API.

> The LLM handles conversation flow, user intent recognition, and query resolution.

---

### 🔹 2. **Prompting Strategy**

Uses a **ReAct (Reasoning + Acting)** prompt format:

```text
Thought → Action → Action Input → Observation → Final Answer
```

This allows the model to:

* Think through the user query
* Choose tools (like retrieval or search)
* Fetch relevant results
* Provide informed responses

---

### 🔹 3. **Embeddings**

* Embeddings are generated using **Sentence Transformers**.
* These are used to index hotel documents in the vector database.
* Supports semantic similarity search for contextual queries.

---

### 🔹 4. **Vector Store: ChromaDB**

* Used to store embedded documents.
* Enables the RAG mechanism by retrieving contextually relevant chunks for the LLM.
* Offers fast similarity search at scale.

---

### 🔹 5. **Integrated Tools**

| Tool                        | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| **Retriever**               | Pulls data from embedded hotel documents |
| **Online Search (SerpAPI)** | Fetches real-time info when needed       |

---

### 🔹 6. **Chat Memory**

Maintains conversation history using LangChain memory to:

* Track previous queries
* Maintain natural multi-turn interactions
* Improve coherence across the conversation

---

## ▶️ Running the App

1. **Set up your `.env` file** at the root level:

```
OPENAI_API_KEY=your_openai_key
SERPAPI_API_KEY=your_serpapi_key
```

2. **Navigate to the `src/` folder**:

```bash
cd src
```

3. **Launch the app with Streamlit**:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```bash
├── src/                # Main app code (LangChain logic, agent, etc.)
├── data/               # Raw & processed data
├── chroma_db/          # Vector DB for storing hotel embeddings
├── notebook/           # EDA and experiment notebooks
└── .env                # API keys and config
```

---

## 🚀 Future Improvements

* Add more metadata.
* Support for multilingual queries.

---

