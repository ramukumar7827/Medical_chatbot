# 🏥 Medical Chatbot using RAG and LLMs

An AI-powered medical chatbot that answers health-related queries using **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**. The system retrieves relevant medical documents and generates accurate, context-aware responses in real time.
<img width="1008" height="869" alt="image" src="https://github.com/user-attachments/assets/3d8e12c8-d275-4354-8710-ac411587f8f5" />

---

##  Features
- AI-powered chatbot for medical question answering
- Retrieval-Augmented Generation (RAG) for accurate responses
- Semantic search using vector embeddings
- Context-aware answers based on medical documents
- Dockerized deployment on AWS EC2

---

## Tech Stack
- **Programming Language:** Python  
- **Frameworks & Libraries:** LangChain, Flask  
- **LLM:** Hugging Face Transformers  
- **Vector Database:** Pinecone  
- **Embedding Model:** Hugging Face Embeddings  
- **Deployment:** Docker, AWS EC2  

---

##  Architecture
1. Medical documents are processed and converted into embeddings.
2. Embeddings are stored in **Pinecone** for semantic search.
3. User queries are converted into embeddings.
4. Relevant document chunks are retrieved from Pinecone.
5. Retrieved context is passed to an **LLM** to generate accurate responses.

---
