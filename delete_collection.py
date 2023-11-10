from dotenv import load_dotenv
import chromadb
import os

load_dotenv()
client = chromadb.PersistentClient(path=os.environ.get("INDEX_NAME"))
for  cols in client.list_collections():
    
    if cols.name =="GRI":
        print(cols)
        client.delete_collection(name=cols.name)
