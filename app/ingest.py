import glob
import os

from constants import INDEX_NAME
from dotenv import load_dotenv
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from load_pdf import load_pdf
from pinecone import Pinecone
from pinecone.db_control.models import ServerlessSpec

load_dotenv()


DATA_DIR = "../data"


def create_or_get_index(index_name: str):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(name=index_name)


# The vectorstore to use to index the child chunks
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index = create_or_get_index(INDEX_NAME)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)


def load_docs():
    docs = []
    for path in glob.glob(f"{DATA_DIR}/*"):
        if path.lower().endswith(".pdf"):
            load_pdf(path, retriever, id_key)
        # elif path.lower().endswith(".pptx"):
        #     docs.extend(UnstructuredPowerPointLoader(path).load())
        # else:
        #     docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


def main():
    load_docs()
    # print(f"Loaded {len(docs)} documents")
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    # )
    # chunks = splitter.split_documents(docs)
    # print(f"Split into {len(chunks)} chunks")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # # integrate with pinecone
    # index = create_or_get_index(INDEX_NAME)
    # vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    # vector_store.add_documents(chunks)
    # print(f"Indexed {len(chunks)} chunks → {INDEX_NAME}")


if __name__ == "__main__":
    main()
