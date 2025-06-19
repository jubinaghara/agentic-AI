import os
import gc
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

# Load environment variables
load_dotenv()

# Setup embeddings client
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
)

# Sample document
text = """
VACATION POLICY: All full-time employees get 15 vacation days per year.
Part-time employees get proportional days. Vacation must be requested 2 weeks in advance.
"""

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
documents = [Document(page_content=chunk, metadata={"source": "policy"}) for chunk in chunks]

# Setup Azure Search vector store
vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
    index_name="rag-demo-index",
    embedding_function=embedding_model
)

# Add documents to vector store
vector_store.add_documents(documents)

# Setup chat model (gpt-35-turbo)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0.3
)

# User question
question = "How many vacation days are employees allowed?"

# Retrieve relevant chunks
relevant_docs = vector_store.similarity_search(question, k=3)
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Prepare chat messages
messages = [
    SystemMessage(content="You are a helpful assistant for company policies."),
    HumanMessage(content=f"""Based on the following company policy context, answer the question:

Context:
{context}

Question: {question}""")
]

# Format messages properly
prompt = ChatPromptValue(messages=messages)

# Generate answer
response = llm.invoke(prompt)

# Print results
print("\n🎯 Answer:")
print(response.content)

print("\n📄 Retrieved Context:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Chunk {i}: {doc.page_content}")

# Clean up resources to prevent shutdown errors
del vector_store  # Explicitly delete vector store first
gc.collect()      # Force garbage collection
