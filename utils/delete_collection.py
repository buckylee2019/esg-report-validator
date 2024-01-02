from dotenv import load_dotenv
import chromadb
import os
import sys

load_dotenv()
client = chromadb.PersistentClient(path=os.environ.get("INDEX_NAME"))
for  cols in client.list_collections():
    

    print(cols)
    client.delete_collection(name=cols.name)
