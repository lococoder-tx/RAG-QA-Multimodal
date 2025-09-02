import glob

from app.load_pdf import load_pdf
from app.retriever import setup_retriever

DATA_DIR = "./data"


# Set up the retriever
_retriever = setup_retriever()
id_key = "doc_id"


def load_docs():
    docs = []
    for path in glob.glob(f"{DATA_DIR}/*"):
        if path.lower().endswith(".pdf"):
            load_pdf(path, _retriever, id_key)
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
