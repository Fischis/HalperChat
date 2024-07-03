import streamlit as st
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.chains import ConversationChain
from langchain.memory import (ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationKGMemory)
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter
from bs4 import BeautifulSoup
import requests
import json

from langchain.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import VectorDBQA, VectorDBQAWithSourcesChain

#
#
#
st.title('🦜🔗 Hilfe Hilfe')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
if 'vStore' not in st.session_state:
    st.session_state.vStore = None


#llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo-instruct')

#
# 
#

def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)

def ScrapePagesToVectorDB():
    qaPages = []
    qaUrls = []
    with open('URLS', 'r') as file:
        for line in file:
                url = line.strip().strip(',').strip('"')
                if url:  
                    qaUrls.append(url)

    for url in qaUrls:
        qaPages.append({'text': extract_text_from(url), 'source': url})

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs, metadatas = [], []
    for page in qaPages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")
    #
    store = FAISS.from_texts(docs, OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key), metadatas=metadatas)
    store.save_local("faiss_store1")    

def ReadVectorDB ():
    tmpStore = FAISS.load_local("faiss_store1", OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key), allow_dangerous_deserialization=True)
    st.sidebar.info (tmpStore.index.ntotal)
    return tmpStore
    #all_docs = tmpStore.index_to
    # _docs.values()
    #all_metadatas = tmpStore.index_to_metadatas.values()

    # Print all documents and their corresponding metadata
    #for doc, metadata in zip(all_docs, all_metadatas):
    #    print(f"Document: {doc}, Metadata: {metadata}")
       
#
#
#


def GenerateResponse (userInput):
    template = """Du bist ein Support-Chatbot. Deine Aufgabe ist es, den Benutzern bei ihren Anliegen zu helfen. Antworte höflich und informativ auf die folgenden Nachrichten.
        Chatverlauf: {history}
        Benutzer: {input}
        Support-Chatbot:"""
  
    llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0, openai_api_key=openai_api_key)
    PROMPT = PromptTemplate(input_variables=["history","input"], template=template)
    myMemory=ConversationBufferMemory()
    #cChain = ConversationChain(llm=llm, memory=myMemory, prompt = PROMPT)
    
    cChain = VectorDBQAWithSourcesChain.from_llm(llm=llm, vectorstore=st.session_state.vStore)   
    result = cChain({"question": userInput})
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    st.info(result)

#
#
#
if (st.sidebar.button('Generate Vector DB')):
    ScrapePagesToVectorDB()

if (st.sidebar.button('Load Vector DB')):
    st.session_state.vStore = ReadVectorDB()
    st.sidebar.info (st.session_state.vStore.index.ntotal)

#
#
#
with st.form('my_form'):
    text = st.text_area('Wie kann ich Ihnen helfen?', '')
    submitted = st.form_submit_button('Go')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        GenerateResponse(text)



