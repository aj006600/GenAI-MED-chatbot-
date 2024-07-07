'''
conda activate pt2_1
cd desktop/GenAI/main/MED-chatbot
streamlit run MED-chatbot_streamlit_demo.py
'''
import os, tempfile
from pathlib import Path
from VectorDB import vectordb_
from Local_LLM import load_vectordb_, local_llm_
from Virtual_Streamer import vs

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

import streamlit as st
st.set_page_config(page_title="MED-chatbot")
st.title("MED-chatbot")

cur_dir = os.path.dirname(__file__)
LOCAL_VECTOR_STORE_DIR = os.path.join(cur_dir, "..", "file", 'db')

def define_openAI():
    openai_api_key = st.sidebar.text_input('key:', type='password')
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
    return llm
def define_local_llm():
    llm = LlamaCpp(
        model_path=os.path.join(cur_dir, "..", "models", "llama3", 
                                "omost-dolphin-2.9-llama3-8b-q4_k_m.gguf"),
        n_gpu_layers=100,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
    )
    return llm

mode = st.sidebar.radio(
    "LLM type: ",
    ('local LLM', 'openAI'))
if mode == 'local LLM':
    llm = define_local_llm()
elif mode == 'openAI':
    llm = define_openAI()

def embeddings_on_local_vectordb():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    db_path = os.path.join(cur_dir, "..", "..", "main", "files", "db")
    vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
    print('vectordb loaded')
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def query_llm(retriever, query, llm):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

def add_prompt(llm, query):
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    init_Prompt = """
    You are a professional medical assistant, well-versed in knowledge of various cancers and chronic diseases. \
    You are eager to help patients and potential patients acquire relevant medical pathology knowledge, \
    and ultimately provide advice on seeking medical treatment. \
    Provide an answer to the following question in about 300 words in Traditional Chinese. Ensure that the answer is informative, relevant, and concise: \
    {query}
    """
    input_prompt = PromptTemplate(input_variables=["query"], template=init_Prompt)
    return LLMChain(prompt=input_prompt, llm=llm)

def query_llm_direct(query, llm):
    llm_chain = add_prompt(llm, query)
    result = llm_chain.invoke({"query": query})
    result = result['text']
    st.session_state.messages.append((query, result))
    return result

# def vedio(response):
#     vs.main(response)
#     return

def boot():
    if "messages" not in st.session_state:
        st.session_state.messages = []    
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    if query := st.chat_input():
        st.chat_message("human").write(query)
        if "retriever" in st.session_state:
            response = query_llm(st.session_state.retriever, query, llm)
        else:
            response = query_llm_direct(query, llm)
        st.chat_message("ai").write(response)
        # vedio(response)

if __name__ == '__main__':
    boot()