import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

def splite(CHUNK_SIZE, CHUNK_OVERLAP):
    cur_dir = os.getcwd()
    folder_path  = os.path.join(cur_dir, "..", "..", "data")
    pdfs = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    PDFS_data = []
    for pdf in pdfs:
        pdf_dir = os.path.join(cur_dir, "..", "..", "data", pdf)
        loader = PyMuPDFLoader(pdf_dir)
        pdf_data = loader.load()
        PDFS_data = PDFS_data + pdf_data

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(PDFS_data)
    print('splite finished')
    return docs

def db_persist(docs):
    cur_dir = os.getcwd()
    persist_directory = os.path.join(cur_dir, "..", "..", "main", "files", "db")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    batch_size = 40000 
    vectordb = None
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        if vectordb is None:
            vectordb = Chroma.from_documents(documents=batch, embedding=embedding, persist_directory=persist_directory)
        else:
            vectordb.add_documents(documents=batch)
    vectordb.persist()
    print("Embedding and storage completed.")
    return 

def vectordb_(CHUNK_SIZE, CHUNK_OVERLAP):
    docs = splite(CHUNK_SIZE, CHUNK_OVERLAP)
    db_persist(docs)
    return

if __name__ == '__main__':
    vectordb_(CHUNK_SIZE=1000, CHUNK_OVERLAP=5)