import streamlit as st
from background import apply_css
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
import os
from templete import bot_template, user_template
# Load environment variables
load_dotenv()

# Set your Hugging Face API token
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(chunks):
    embeddings = HuggingFaceEndpointEmbeddings(
        model="mixedbread-ai/mxbai-embed-large-v1",
        task="feature-extraction",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def conversation_chain(vectorstore):
    retriever = vectorstore.as_retriever(k=5)
    
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        max_new_tokens=150,
        do_sample=False,
        device="cuda"
    )
    
    summarization_llm = HuggingFaceEndpoint(
        repo_id="facebook/bart-large-cnn",
        task="summarization",
        max_new_tokens=200,
        do_sample=False
    )
    
    conversation_summary_buffer = ConversationSummaryBufferMemory(
        llm=summarization_llm,
        max_token_limit=1000,
        memory_key="chat_history"
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=conversation_summary_buffer
    )
    
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Talk to PDFs", page_icon=":books:")
    
    apply_css()
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.chat_history = []
    
    st.header("Chat with multiple PDFs :books:")
    
    user_question = st.text_input("Ask anything to your PDFs")
    
    st.write(user_template.replace("{{MSG}}", "hello bot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your files and press continue", accept_multiple_files=True)
        
        if st.button("Continue"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return
            
            with st.spinner("Thinking..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    st.error("No text extracted from PDFs.")
                    return
                
                st.write(f"Extracted text: {raw_text[:500]}...")  # Show part of the text
                
                chunks = get_text_chunks(raw_text)
                st.write(f"Created {len(chunks)} chunks.")
                
                try:
                    vectorstore = get_vectorstore(chunks)
                    st.session_state.conversation = conversation_chain(vectorstore)
                except Exception as e:
                    st.error(f"Error initializing conversation chain: {str(e)}")
    
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)

if __name__ == '__main__':
    main()