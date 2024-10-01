import streamlit as st
from background import apply_css
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub
from langchain.chains import ConversationalRetrievalChain
from templete import bot_template,user_template
from langchain.chains.conversation.memory import ConversationSummaryMemory


import os

# Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""



#def embed_chunk(chunk):
    #return embeddings_model.embed_documents(chunk)

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        reader=PdfReader(pdf)
        for page in reader.pages:
            
            text+=page.extract_text()+ " "
   
    return text

def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(chunks):
    embeddings=HuggingFaceEndpointEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1",task="feature-extraction",
                                                 huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"])
   

    # Create the FAISS vector store from the text chunks
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def model(vectorstore,question):
    retriever = vectorstore.as_retriever(k=3)
    
    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    #task="text-generation",
    max_new_tokens=100,
    do_sample=False,
    )
    
    #related_docs=retriever.invoke({"input":question})
    
    
    
    '''prompt = ChatPromptTemplate.from_messages("""Answer the following question based only on the provided context:
                                                <context>
                                                {context}
                                                </context>
                                                Question: {input}""")'''
    prompt = hub.pull("rlm/rag-prompt")
        
    
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever,chain_type_kwargs={"prompt":prompt},chain_type="stuff")
    
    answer = qa_chain.invoke({"query": question})
    
    
    
    st.write(answer)
    
def get_conversation_chain(vectorstore):
    retriever = vectorstore.as_retriever(k=10)
    llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    max_new_tokens=150,
    do_sample=False,
    device="cuda"
    )
    
    #prompt = hub.pull("rlm/rag-prompt")
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,memory=memory)
    
    return conversation_chain
        
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    
#def get_conversation_chain(vectorstore):
    
def main():
    load_dotenv()
    
    st.set_page_config(page_title="Talk to PDFs",page_icon=":books:")
    
    apply_css()
    
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
        st.session_state.chat_history = []
    
    st.header("Chat with multiple Pdfs :books:")
    
    user_question=st.text_input("Ask anything to your pdfs")
    
    
    
    st.write(user_template.replace("{{MSG}}","hello bot"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","hello human"),unsafe_allow_html=True)
    
    with st.sidebar:
        
        st.subheader("Your documents")
        pdf_docs=st.file_uploader("Uplod your files and press continue",accept_multiple_files=True)
        
        if st.button("Continue"):
            with st.spinner("Thinking"):
                raw_text=get_pdf_text(pdf_docs)
                chunks=get_text_chunks(raw_text)
                #st.write(chunks)
                
                
                vectorstore=get_vectorstore(chunks)
                
                #model(vectorstore,question)
                st.session_state.conversation=get_conversation_chain(vectorstore)
                
             
                # create a converstation chain
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)
if __name__=='__main__':
    main()
