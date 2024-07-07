import os, sys
from VectorDB import vectordb_
from Local_LLM import load_vectordb_, local_llm_

import warnings
warnings.filterwarnings('ignore')

def main():
    cur_dir = os.getcwd()
    db_path = os.path.join(cur_dir, "..", "files", "db")
    if not os.listdir(db_path):
        print('db not exist')
        vectordb_(CHUNK_SIZE=1000, CHUNK_OVERLAP=5)
        vectordb = load_vectordb_()
        qa = local_llm_(vectordb)
    else:
        vectordb = load_vectordb_()
        print('db exist')
        qa = local_llm_(vectordb)

    query = None
    while query == None:
        query = input("Enter your query: ")
        qa.invoke(query)
        query = None
    return

if __name__ == '__main__':
    main()