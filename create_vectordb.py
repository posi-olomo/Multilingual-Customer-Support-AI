import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # Facebook AI Similarity Search
import os
import getpass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

doc = "data/fake_kb.txt"

# Build FAISS vector from the text files
with open(doc, 'r', encoding='utf-8') as file:
  content = file.read()

  # Split content into chunks 
chunk_size = 1000
overlap = 200

if len(content) > chunk_size:
  texts = []
  for i in range(0, len(content), chunk_size - overlap):
    chunk = content[i:i+chunk_size]
    if chunk.strip(): # add only non-empty chunks 
      texts.append(chunk)
else:
  texts = [content]

# Create a vector store from the chunks
vectorstore = FAISS.from_texts(texts, embeddings)

# Save the vector store
save_path = "data/vectorstore"
vectorstore.save_local(save_path)

print(f"Vector store saved in {save_path}")