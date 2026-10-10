"""
===================================================================
Project Title       : Innomatics Research Labs AI Assistant
Problem Statement   : Students and staff struggle to find accurate 
                      information across scattered institute PDF files.
Proposed Solution   : A multi-version RAG application that evolves from 
                      basic LLM chains (V1) to FAISS retrieval (V2), 
                      ChromaDB metadata (V3), and Multi-Query Reranking (V4).
===================================================================
"""

import os
import numpy as np
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# 1. Page Configuration
st.set_page_config(
    page_title="Innomatics Research Labs AI ASSISTANT", 
    page_icon="🤖", 
    layout="wide"
)

# 2. Prevent Vulkan GPU memory buffer crashes on Windows
os.environ["OLLAMA_NUM_GPU"] = "0"

# 3. Safe Secret / Configuration Retrieval (Option 2 Guard)
OLLAMA_URL = "http://localhost:11434"
try:
    if "OLLAMA_BASE_URL" in st.secrets:
        OLLAMA_URL = st.secrets["OLLAMA_BASE_URL"]
except Exception:
    pass  # Gracefully fallback if secrets.toml is not present locally

# 4. First-time Visitor Animation
if "visited" not in st.session_state:
    st.balloons()
    st.session_state["visited"] = True

INNOMATICS_EMBLEM_SVG = """data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><path fill='%23FF4B4B' d='M20,20 L50,5 L80,20 L80,80 L50,95 L20,80 Z'/><path fill='%23FFFFFF' d='M35,35 L50,25 L65,35 L65,65 L50,75 L35,65 Z'/><circle cx='50' cy='50' r='8' fill='%23FF8C42'/></svg>"""

# Custom Styling
st.markdown(f"""
    <style>
    .stApp {{
        background-color: #0e1117;
        background-image: linear-gradient(rgba(14, 17, 23, 0.94), rgba(14, 17, 23, 0.94)), url('{INNOMATICS_EMBLEM_SVG}');
        background-repeat: no-repeat;
        background-position: center 55%;
        background-size: 320px;
        background-attachment: fixed;
    }}
    .hero-container {{
        background: linear-gradient(135deg, rgba(30, 38, 64, 0.85) 0%, rgba(13, 27, 42, 0.85) 100%);
        backdrop-filter: blur(12px);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        border: 1px solid #2e3a59;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }}
    .hero-logo {{ width: 68px; height: 68px; filter: drop-shadow(0px 4px 10px rgba(255, 75, 75, 0.4)); }}
    .hero-title {{
        font-size: 2.2rem !important; font-weight: 800 !important;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8c42 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem; line-height: 1.2;
    }}
    .hero-subtitle {{ color: #a0aec0; font-size: 0.98rem; margin-bottom: 0px; }}
    .version-badge {{
        background-color: #232d42; color: #63b3ed; padding: 6px 14px;
        border-radius: 20px; font-size: 0.88rem; font-weight: 600;
        border: 1px solid #3182ce; display: inline-block; margin-bottom: 1.2rem;
    }}
    </style>
""", unsafe_allow_html=True)

DATA_PATH = r"C:\Users\crazy\OneDrive\Desktop\INNOMATICS_RAG_PROJECT\data"
CHROMA_PATH = r"C:\Users\crazy\OneDrive\Desktop\INNOMATICS_RAG_PROJECT\chroma_db_minilm"

# Resource Caching
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_documents_and_chunks():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    for chunk in chunks:
        source_path = chunk.metadata.get("source", "")
        chunk.metadata["source"] = os.path.basename(source_path)
        chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1
        chunk.metadata["document_type"] = "Official Institute PDF"
    return chunks

@st.cache_resource
def get_faiss_retriever():
    embeddings = get_embeddings()
    chunks = get_documents_and_chunks()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

@st.cache_resource
def get_chroma_vectorstore():
    embeddings = get_embeddings()
    chunks = get_documents_and_chunks()
    return Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH,
        collection_name="innomatics_minilm_v3"
    )

def multi_query_retrieve(chroma_vs, primary_query):
    base_retriever = chroma_vs.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    sub_queries = [
        primary_query,
        "Module 1 Module 2 Python Programming Data Science syllabus curriculum",
        "Institute contact details phone email branch addresses"
    ]
    all_docs = []
    seen_contents = set()
    for sub_q in sub_queries:
        docs = base_retriever.invoke(sub_q)
        for doc in docs:
            if doc.page_content not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc.page_content)
    return all_docs

def rerank_documents(query, candidate_docs, embeddings, top_n=8):
    query_embedding = embeddings.embed_query(query)
    doc_texts = [doc.page_content for doc in candidate_docs]
    doc_embeddings = embeddings.embed_documents(doc_texts)
    scores = []
    for doc, doc_emb in zip(candidate_docs, doc_embeddings):
        similarity = np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        doc.metadata["relevance_score"] = float(similarity)
        scores.append((similarity, doc))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scores[:top_n]]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Sidebar Controls
st.sidebar.markdown("## ⚙️ Pipeline Control")
selected_version = st.sidebar.selectbox(
    "Choose System Architecture:",
    ("Version 1 - LangChain Basics (No RAG)", "Version 2 - RAG with FAISS", "Version 3 - RAG with ChromaDB", "Version 4 - RAG with Reranking")
)

st.sidebar.divider()
st.sidebar.caption("🚀 Powered by Local Llama 3.2 & LangChain")

# Header Section
st.markdown(f"""
    <div class="hero-container">
        <img src="{INNOMATICS_EMBLEM_SVG}" class="hero-logo" alt="Innomatics Emblem" />
        <div>
            <div class="hero-title">Innomatics Research Labs AI ASSISTANT</div>
            <p class="hero-subtitle">Your official, intelligent campus concierge for syllabi, admissions, branch details, and inquiries.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="version-badge">Active Engine: <b>{selected_version}</b></div>', unsafe_allow_html=True)

# LLM Engine Initialization
llm = ChatOllama(
    model="llama3.2",
    temperature=0.2,
    base_url=OLLAMA_URL,
    num_gpu=0
)

output_parser = StrOutputParser()

student_prompt_template = """You are the official Institute AI Assistant for Innomatics Research Labs. Your role is to provide accurate, polite, and helpful information to students, faculty, and visitors.

Guidelines:
1. Answer the question clearly, concisely, and professionally using ONLY the official information provided below.
2. NEVER mention phrases like "based on the provided documents", "according to the context", "in the files", or "uploaded text". Speak naturally as a representative of Innomatics Research Labs.
3. If the specific information is not available, politely state that the detail is currently not listed and guide the user to contact their respective branch administration or department coordinator for assistance.

Official Information:
{context}

User Question: 
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(student_prompt_template)
query = st.text_input("💬 Ask a question about courses, fees, or branches:", placeholder="e.g., What are the courses offered by innomatics research labs?")

if st.button("🚀 Ask Assistant"):
    if not query.strip():
        st.warning("⚠️ Please enter a question before submitting.")
    else:
        try:
            if "Version 1" in selected_version:
                v1_prompt = ChatPromptTemplate.from_template("You are an AI assistant for Innomatics Research Labs. Answer: {question}")
                v1_chain = v1_prompt | llm | output_parser
                with st.spinner("🤖 Consulting Llama 3.2 Knowledge (V1)..."):
                    response = v1_chain.invoke({"question": query})
                    st.markdown("### 💡 Answer (V1 - Standard Llama 3.2)")
                    st.info(response)

            elif "Version 2" in selected_version:
                with st.spinner("🔍 Querying FAISS Vector Store (V2)..."):
                    retriever = get_faiss_retriever()
                    v2_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | output_parser)
                    response = v2_chain.invoke(query)
                    st.markdown("### 💡 Answer (V2 - FAISS Context)")
                    st.success(response)

            elif "Version 3" in selected_version:
                with st.spinner("🏷️ Searching ChromaDB Vector Store with Metadata (V3)..."):
                    chroma_vs = get_chroma_vectorstore()
                    retriever = chroma_vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                    v3_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | output_parser)
                    response = v3_chain.invoke(query)
                    st.markdown("### 💡 Answer (V3 - ChromaDB Context)")
                    st.success(response)

            elif "Version 4" in selected_version:
                with st.spinner("🎯 Decomposing Query & Reranking Results (V4)..."):
                    chroma_vs = get_chroma_vectorstore()
                    embeddings = get_embeddings()
                    candidate_docs = multi_query_retrieve(chroma_vs, query)
                    top_reranked_docs = rerank_documents(query, candidate_docs, embeddings, top_n=8)
                    context_text = format_docs(top_reranked_docs)
                    v4_chain = prompt | llm | output_parser
                    response = v4_chain.invoke({"context": context_text, "question": query})
                    st.markdown("### 💡 Answer (V4 - Reranked Context)")
                    st.success(response)
        except Exception as e:
            st.error(f"⚠️ Execution Error: {str(e)}")