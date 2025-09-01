from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.constants import INDEX_NAME, SYSTEM_PROMPT
from app.ingest import create_or_get_index

load_dotenv()

# Store for conversation sessions
store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template(
            """{query}, provided this context: {context}"""
        ),
    ]
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

    llm = ChatOpenAI(
        model="gpt-5",
        temperature=0,
        reasoning={"effort": "minimal"},
        use_responses_api=True,
        output_version="responses/v1",
    )

    # Build the RAG chain using LCEL
    def retrieve_and_format(query: str) -> str:
        docs = retriever.invoke(query)
        return format_context(docs)

    # Create the RAG chain
    rag_chain = ANSWER_PROMPT | llm | StrOutputParser()

    # Wrap with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="history",
    )

    session_id = "default_session"
    print("Ask me about your docs. Type 'exit' to quit.")

    while True:
        q = input("\n> ")
        if q.strip().lower() in ("exit", "quit"):
            break

        try:
            response = conversational_rag_chain.invoke(
                {"query": q, "context": retrieve_and_format(q)},
                config={
                    "configurable": {"session_id": session_id},
                },
            )
            print("\n" + response)
            # print("session history: ", get_session_history(session_id))
        except (ValueError, RuntimeError, ConnectionError) as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
