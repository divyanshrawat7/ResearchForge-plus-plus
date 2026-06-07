import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

st.set_page_config(page_title="ResearchForge++", layout="wide")

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
llm = genai.GenerativeModel("gemini-2.5-flash")

@st.cache_resource
def load_resources():
    df = pd.read_json("data/papers.json")
    index = faiss.read_index("vector_db/researchforge_faiss.index")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, index, model

df, faiss_index, embedding_model = load_resources()

def retrieve_papers(query, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )
    papers = []
    for idx in indices[0]:
        papers.append(df.iloc[idx].to_dict())
    return papers

def build_context(papers):
    context = ""
    for paper in papers:
        context += f"Title: {paper.get('title','')}\nAbstract: {paper.get('abstract','')}\n\n"
    return context

def run_agent(query, papers, task_prompt):
    context = build_context(papers)
    prompt = f"""
User Question:
{query}

Scientific Evidence:
{context}

Task:
{task_prompt}

Use only the retrieved evidence.
"""
    return llm.generate_content(prompt).text

st.title("ResearchForge++")
query = st.text_area("Research Question")

tabs = st.tabs([
    "Method Comparison",
    "Contradictions",
    "Research Gaps",
    "Research Ideas",
    "Debate"
])

if query:
    papers = retrieve_papers(query)

    with tabs[0]:
        if st.button("Run Method Comparison"):
            st.write(run_agent(query, papers,
                "Compare methods, datasets, metrics, strengths and limitations."))

    with tabs[1]:
        if st.button("Run Contradiction Analysis"):
            st.write(run_agent(query, papers,
                "Identify contradictions and conflicting findings."))

    with tabs[2]:
        if st.button("Run Gap Detection"):
            st.write(run_agent(query, papers,
                "Identify research gaps and underexplored directions."))

    with tabs[3]:
        if st.button("Generate Research Ideas"):
            st.write(run_agent(query, papers,
                "Generate realistic research ideas with implementation and evaluation plans."))

    with tabs[4]:
        if st.button("Run Debate"):
            st.write(run_agent(query, papers,
                "Create an evidence-based debate and final conclusion."))
