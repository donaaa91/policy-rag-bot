import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- UPDATED IMPORTS FOR LANGCHAIN 1.2.7 ---
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# --------------------------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="The Policy Analyst", layout="wide")
st.title("AI Policy Analyst")

# SIDEBAR FOR API KEY
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# ---------------------------------------------------------
# LOGIC 1: THE BRAIN
# ---------------------------------------------------------
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={"device": "cpu"})

# ---------------------------------------------------------
# LOGIC 2: THE INGESTION ENGINE
# ---------------------------------------------------------
def process_pdf(uploaded_file):
    # save file temporarily so the loader can read it
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # load the pdf text
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # split it into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    # create the vector store
    vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
    return vectors

# ---------------------------------------------------------
# UI: FILE UPLOAD
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload your Policy Document (PDF)", type="pdf")

if uploaded_file and api_key:
    if "vectors" not in st.session_state:
        with st.spinner("Analyzing document... (This runs on CPU, might take a moment)"):
            st.session_state.vectors = process_pdf(uploaded_file)
        st.success("Document Analyzed! Ask your questions below.")

    # ---------------------------------------------------------
    # THE LOGIC
    # ---------------------------------------------------------
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        If the answer is not in the context, say "I don't see that in the document."

        <context>
        {context}
        </context>

        Question: {input}
        """
    )

    # Creating the retrieval chain using LCEL (LangChain Expression Language)
    retriever = st.session_state.vectors.as_retriever()
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Build the chain using modern LCEL syntax
    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # User Input
    user_prompt = st.text_input("Ask a question about the policy:")

    if user_prompt:
        # Run the chain
        answer = chain.invoke(user_prompt)
        
        # Get context docs for citations
        context_docs = retriever.invoke(user_prompt)

        # Display Answer
        st.write("### 🤖 Answer:")
        st.write(answer)

        # CITATIONS FEATURE
        with st.expander("🔎 View Source / Evidence"):
            for i, doc in enumerate(context_docs):
                st.write(f"**Source Chunk {i + 1} (Page {doc.metadata.get('page', 0) + 1}):**")
                st.write(doc.page_content)
                st.divider()

elif uploaded_file and not api_key:
    st.warning("Please enter your Groq API Key in the sidebar to activate the AI.")