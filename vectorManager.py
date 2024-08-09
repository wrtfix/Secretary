import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

class VectorManager(object):

    def __init__(self, embeddings=OllamaEmbeddings(model="nomic-embed-text")):
        self.embeddings = embeddings

    def save_faiss(self, documents):
        
        print('Generando vectores... ')
        self.faiss = FAISS.from_documents(documents[0], self.embeddings)

        print('Guardando vectores...')
        self.faiss.save_local(os.getenv("VECTOR_FOLDER"), "VectorFile")

        return self.faiss
        
    def load_faiss(self):
        print('Cargando vectores...')
        if (self.faiss == None):
            self.faiss = FAISS.load_local(os.getenv("VECTOR_FOLDER"), self.embeddings, index_name="VectorFile", allow_dangerous_deserialization=True)
        print('Vectores cargados...')
        return self.faiss
