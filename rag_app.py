import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

key=os.getenv('google_api_key')
genai.configure(api_key=key)

model=genai.GenerativeModel('models/gemini-2.5-flash-lite')
def load_embedding():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
with st.spinner('loading embedding model'):
    embedding_model=load_embedding()

st.set_page_config(page_title="RAG Demo",page_icon=":robot:",layout="wide")
st.title('RAG Assistant :blue[Using Embeddings and LLM]⚖️')

st.subheader(':green[Your Intelligent Document Assistant]📄')
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    pdf=PdfReader(uploaded_file)
    
    raw_text=""
    for page in pdf.pages:
        raw_text+=page.extract_text()
    if raw_text.strip():
        doc=Document(page_content=raw_text)
        
        splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        
        chunks_text=splitter.split_documents([doc])
        
        text =[i.page_content for i in chunks_text]
        
        vector_db=FAISS.from_texts(text,embedding_model)
        
        retrive= vector_db.as_retriever()
        
        st.success('document processed and save scussfully !! ask a question now')
        
        query=st.text_input('ask me a question ')
        
        if query:
            with st.chat_message('human'):
                
                with st.spinner('Analyzing the document.....'):
                
                    relevant_data=retrive.invoke(query)
                
                    content='\n\n'.join([i.page_content for i in relevant_data])
                
                    prompt=f'''
                    you are an AI expert use the generated  content
                    {content} to answer the query {query}.if you are not
                    sure with the answer say "I have no content related 
                    to this questions.please ask relavent query to answer"
                    
                    Result in bullet points'''
                    
                    response=model.generate_content(prompt)
                    
                    st.markdown('## :green[results ]')
                    st.write(response.text)
    else:
        st.warning('Drop the file in PDF format')