import base64

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.constants import SYSTEM_PROMPT
from app.retriever import setup_retriever

load_dotenv()

# Store for conversation sessions
store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ANSWER_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
#         MessagesPlaceholder(variable_name="history"),
#         HumanMessagePromptTemplate.from_template(
#             """{query}, provided this context: {context}"""
#         ),
#     ]
# )


def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        if doc.metadata.get("type") == "image":
            b64.append(doc)
        else:
            text.append(doc)

    print(f"Found {len(b64)} images and {len(text)} texts")
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["query"]

    context_text = ""
    for i, text_element in enumerate(docs_by_type["texts"]):
        context_text += f"[{i}] {text_element.page_content}\n(source: {text_element.metadata.get('source', 'unknown')}, p.{text_element.metadata.get('page', None)})\n"

    prompt_template = f"""
    Answer the question based on the following context, which can include text, tables, and the below image:
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    for image in docs_by_type["images"]:
        prompt_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image.page_content}"},
            }
        )

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_content),
        ]
    )


def main():
    retriever = setup_retriever(k=3)

    llm = ChatOpenAI(
        model="gpt-5",
        temperature=0,
        reasoning={"effort": "minimal"},
        use_responses_api=True,
        output_version="responses/v1",
    )

    # Create the RAG chain
    # TODO: Adapt for usage with runnable with message history
    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "query": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | llm
        | StrOutputParser()
    )

    # # Wrap with message history
    # conversational_rag_chain = RunnableWithMessageHistory(
    #     rag_chain,
    #     get_session_history,
    #     input_messages_key="query",
    #     history_messages_key="history",
    # )

    # session_id = "default_session"
    print("Ask me about your docs. Type 'exit' to quit.")

    while True:
        q = input("\n> ")
        if q.strip().lower() in ("exit", "quit"):
            break

        try:
            response = chain.invoke(q)
            print("\n" + response)
        except (ValueError, RuntimeError, ConnectionError) as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
