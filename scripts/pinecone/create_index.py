import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Carrega .env
load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX")
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# Inicializa cliente Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Lista índices existentes
existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if INDEX_NAME in existing_indexes:
    print(f"✅ Index '{INDEX_NAME}' já existe")
else:
    print(f"🚀 Criando index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,          # OpenAI text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"✅ Index '{INDEX_NAME}' criado com sucesso")
