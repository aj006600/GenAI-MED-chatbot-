import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate

cur_dir = os.getcwd()

def load_vectordb_():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    db_path = os.path.join(cur_dir, "..", "..", "main", "files", "db")
    vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
    print('vectordb loaded')
    return vectordb

def llm_():
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

# def llm_bot(llm):
#     DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
#         input_variables=["question"],
#         template="""<<SYS>> 
#             You are a professional medical assistant, well-versed in knowledge of various cancers and chronic diseases. \
#             You are eager to help patients and potential patients acquire relevant medical pathology knowledge, \
#             and ultimately provide advice on seeking medical treatment.
#             <</SYS>> 
        
#         [INST] Provide an answer to the following question in 300 words. Ensure that the answer is informative, \
#                 relevant, and concise:
#                 {question} 
#         [/INST]""",
#     )
#     DEFAULT_SEARCH_PROMPT = PromptTemplate(
#         input_variables=["question"],
#         template="""You are a professional medical assistant, well-versed in knowledge of various cancers and chronic diseases. \
#             You are eager to help patients and potential patients acquire relevant medical pathology knowledge, \
#             and ultimately provide advice on seeking medical treatment. \
#             Provide an answer to the following question in about 300 words. Ensure that the answer is informative, \
#             relevant, and concise: \
#             {question}""",
#     )
#     QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
#         default_prompt=DEFAULT_SEARCH_PROMPT,
#         conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
#     )
#     prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)
#     return prompt

def qa_bot(vectordb, llm):
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        verbose=True
    )
    return qa

def local_llm_(vectordb):
    llm = llm_()
    qa = qa_bot(vectordb, llm)
    return qa

if __name__ == '__main__':
    vectordb = load_vectordb_()
    qa = local_llm_(vectordb)
