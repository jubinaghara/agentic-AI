import os
import pymongo
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# MongoDB setup
client = pymongo.MongoClient(os.getenv("MONGO_DB_URL"))
db = client.sample_mflix
collection = db.movies

# Hugging Face authentication
hugging_face_token = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
if hugging_face_token is None or not hugging_face_token.strip():
    raise ValueError("HUGGING_FACE_ACCESS_TOKEN is not set in the environment.")

login(token=hugging_face_token)

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def generate_embedding(text: str) -> list[float]:
    print("Generating embedding for:", text)
    embedding = model.encode(text)
    return embedding.tolist()

# Example usage
print(generate_embedding("Hello word is good a world"))
