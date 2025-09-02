import os

from dotenv import load_dotenv
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.storage import RedisStore
from langchain_community.utilities.redis import get_client
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone.db_control.models import ServerlessSpec

from .constants import INDEX_NAME

load_dotenv()


def create_or_get_index(index_name: str):
    """Create or get an existing Pinecone index."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(name=index_name)


def setup_retriever(
    index_name: str = INDEX_NAME,
    redis_url: str = "redis://localhost:6379",
    embedding_model: str = "text-embedding-3-small",
    id_key: str = "doc_id",
    *,
    k: int = 3,
):
    """
    Set up and return a MultiVectorRetriever with Pinecone vectorstore and Redis storage.

    Args:
        index_name: Name of the Pinecone index
        redis_url: Redis connection URL
        embedding_model: OpenAI embedding model to use
        id_key: Key to use for document IDs

    Returns:
        MultiVectorRetriever: Configured retriever instance
    """
    # The vectorstore to use to index the child chunks
    embeddings = OpenAIEmbeddings(model=embedding_model)
    index = create_or_get_index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    # The storage layer for the parent documents
    store = RedisStore(client=get_client(redis_url))

    # The retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": k},
    )

    return retriever
