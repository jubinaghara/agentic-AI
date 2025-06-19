#This code loads company policy, splits it into small pieces (chunks), converts those pieces into embeddings (which are special numerical representations of text that capture meaning), stores them in Azure Search, and then lets you ask questions in a chat loop. When you ask a question, it finds the most relevant policy pieces using embeddings, sends them to an AI model to generate an answer, and shows you both the answer and the source.
#Embeddings are vectors (lists of numbers) that represent the meaning of text, allowing computers to compare and find similar content based on meaning, not just exact words.
#The AI model’s role in code is to take the most relevant company policy text (found using embeddings) and generate a clear, human-like answer to your question, making it easy for employees to get instant, understandable explanations of company rules.

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
Embargoed Jurisdictions:
The exportation, reexportation, sale or supply, directly or indirectly, from the United States, or by
a U.S. person wherever located, of any Sophos goods, software, technology (including technical data), or services to Iran, Syria, North Korea, Cuba, or the Donetsk, Luhansk & Crimea region of Ukraine is strictly prohibited without prior authorization by the U.S. Government.

Prohibited and/or Restricted Person Lists:
Sophos products may not be sold, exported, or reexported to any person or entity designated as prohibited or restricted by the United States, United Kingdom, or European Union (including, but not limited to the U.S. Treasury Department’s list of Specially Designated Nationals or the U.S. Department of Commerce Denied Person’s List or Entity List).

Prohibited Uses:
Sophos products may not be used for (i) military purposes, or (ii) use in connection with the development, production, handling, operation, maintenance, storage, detection, identification or dissemination of chemical, biological or nuclear weapons, or other nuclear explosive devices, or the development, production, maintenance or storage of missiles capable of delivering such weapons.
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

print("🤖 Company Policy Assistant (type 'exit' or 'quit' to stop)\n")

while True:
    question = input("You: ").strip()
    if question.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

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

    print("\n🎯 Answer:")
    print(response.content)

    print("\n📄 Retrieved Context:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Chunk {i}: {doc.page_content}")
    print("\n" + "-"*60 + "\n")

# Clean up resources to prevent shutdown errors
del vector_store
gc.collect()
