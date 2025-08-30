from pprint import pprint

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.constants import INDEX_NAME, SYSTEM_PROMPT
from app.ingest import create_or_get_index

load_dotenv()


ANSWER_PROMPT = PromptTemplate.from_template(
    """{system}

Question: {question}

Context:
{context}

Answer (with citations):
"""
)


def format_context(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", None)
        cite = f"{src}" + (f", p.{page+1}" if page is not None else "")
        lines.append(f"[{i}] {d.page_content}\n(source: {cite})")
    return "\n\n".join(lines)


def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index = create_or_get_index(INDEX_NAME)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(model="gpt-5", temperature=0, reasoning={"effort": "minimal"})

    print("Ask me about your docs. Type 'exit' to quit.")

    prev_resp = None
    while True:
        q = input("\n> ")
        if q.strip().lower() in ("exit", "quit"):
            break

        docs = retriever.invoke(q)
        ctx = format_context(docs)
        prompt = ANSWER_PROMPT.format(system=SYSTEM_PROMPT, question=q, context=ctx)

        prev_id = prev_resp.response_metadata.get("id", None) if prev_resp else None
        resp = llm.invoke(prompt, previous_response_id=prev_id)

        print("\n" + resp.content[0].get("text"))
        prev_resp = resp


if __name__ == "__main__":
    main()
