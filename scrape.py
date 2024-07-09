
import os
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import json


try:
    openai_api_key = os.environ.get("OPEN_AI_KEY")
except:
    print ("Error: set OPEN_AI_KEY in Environment")



def FetchLinks(url):
    try:
        # Anfrage an die URL senden
       

        response = requests.get(url)
        response.raise_for_status()  # Stellt sicher, dass keine HTTP-Fehler auftreten
        print (url)

        soup = BeautifulSoup(response.text, 'html.parser')

        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
        
        urls = []        
        for link in links:
            if  link.startswith("https://hilfe.web.de"):
                urls.append(f"{link}")

        return urls

    except requests.RequestException as e:
        print(f"Fehler beim Abrufen der Webseite: {e}")
        return None



def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)

def ScrapePagesToVectorDB(baseUrl, storeName):
    qaPages = []
    qaUrls_1 = FetchLinks(baseUrl)
    qaUrls_2 = []

    print ("******************************")
    print (len (qaUrls_1))
    for url in qaUrls_1:
        tmpLinks = FetchLinks (url)
        if tmpLinks is None: 
            qaUrls_1.remove (url)
        else:
            print ('.', end=' ')
            qaUrls_2.extend (tmpLinks)

    print ("******************************")
    print (len (qaUrls_2))

    qaUrls = qaUrls_1 + qaUrls_2
    
    for url in qaUrls:
        if url is not "https://hilfe.web.de/":        
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
    store.save_local(storeName) 
    print ("Store Saved")   
 


#print (FetchLinks("https://hilfe.web.de/"))

ScrapePagesToVectorDB('https://hilfe.web.de/', "faiss_store4")