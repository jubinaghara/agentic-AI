import os
import pymongo
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# MongoDB setup with SSL fix
client = pymongo.MongoClient(
    os.getenv("MONGO_DB_URL"),
    tls=True,
    tlsInsecure=True
)
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
    return embedding.tolist()  # Fix: changed from .json() to .tolist()


# This code will see if the plot exist and generate the embedding for the plot.
# The exisitng collection of movies is updates and one new field called "plot_embedding_hf" is added to that plot
# This operation is limited to 50.
# for doc in collection.find({'plot':{"$exists": True}}).limit(50):
#    doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#    collection.replace_one({'_id': doc['_id']}, doc)


query = "bavarian widow characters from outer space at war"

results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": generate_embedding(query),
    "path": "plot_embedding_hf",
    "numCandidates": 100,
    "limit": 1,
    "index": "PlotSemanticSearch",
      }}
]);

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')