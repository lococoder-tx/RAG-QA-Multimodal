import time
import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from unstructured.partition.pdf import partition_pdf


def summarize_text_or_table(data):
    # Prompt
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatOpenAI(temperature=0.5, model="gpt-5-chat-latest")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    return _run_with_rate_limit_retries(summarize_chain, data)


def summarize_images(images_b64):
    # Prompt
    prompt_template = """Describe the image in detail. For context,
                  the image is part of a research paper explaining the transformers
                  architecture. Be specific about graphs, such as bar plots."""
    messages = [
        ("system", prompt_template),
        (
            "user",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_b64}"},
                },
            ],
        ),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    model = ChatOpenAI(temperature=0.5, model="gpt-5-chat-latest")
    summarize_chain = prompt | model | StrOutputParser()
    inputs = [{"image_b64": img_b64} for img_b64 in images_b64]
    return _run_with_rate_limit_retries(summarize_chain, inputs)


def _run_with_rate_limit_retries(
    runnable, inputs, batch_size: int = 8, sleep_seconds: int = 60
):
    """
    Execute runnable.batch over inputs with retry on rate limit.

    - Processes in chunks so we can resume from the point of failure.
    - Waits sleep_seconds (default 60s) before retrying the failed chunk.
    """
    if not inputs:
        return []

    results = []
    index_start = 0
    total = len(inputs)

    while index_start < total:
        index_end = min(index_start + batch_size, total)
        sub_inputs = inputs[index_start:index_end]
        try:
            sub_results = runnable.batch(sub_inputs)
            results.extend(sub_results)
            index_start = index_end
        except Exception as e:
            message = str(e).lower()
            if (
                "rate limit" in message
                or "429" in message
                or "too many requests" in message
            ):
                print(
                    f"Rate limit hit at items {index_start}-{index_end-1}. "
                    f"Retrying after {sleep_seconds}s..."
                )
                time.sleep(sleep_seconds)
                # Retry the same sub-batch
                continue
            raise

    return results


def create_docs(
    texts,
    tables,
    images,
    text_summaries,
    table_summaries,
    image_summaries,
    retriever,
    id_key,
):
    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    print(f"Adding {len(texts)} texts")
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]})
        for i, summary in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    print(f"Adding {len(tables)} tables")
    summary_tables = [
        Document(page_content=summary, metadata={id_key: table_ids[i]})
        for i, summary in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in images]
    print(f"Adding {len(images)} images")
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]})
        for i, summary in enumerate(image_summaries)
    ]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, images)))


# Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
def load_pdf(file_path: str, retriever: MultiVectorRetriever, id_key: str):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,  # extract tables
        strategy="hi_res",  # mandatory to infer tables
        extract_image_block_types=[
            "Image"
        ],  # Add 'Table' to list to extract image of tables
        # image_output_dir_path=OUTPUT_PATH,  # if None, images and tables will saved in base64
        extract_image_block_to_payload=True,  # if true, will extract base64 for API usage
        chunking_strategy="by_title",  # or 'basic'
        max_characters=10000,  # defaults to 500
        combine_text_under_n_chars=2000,  # defaults to 0
        new_after_n_chars=6000,
    )

    tables = []
    texts = []
    images_b64 = []

    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Text" in str(type(el)):
                    texts.append(el)
                elif "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)

    print("Summarize texts...")
    text_summaries = summarize_text_or_table(texts)
    print("Summarize tables...")
    table_summaries = summarize_text_or_table(
        [table.metadata.text_as_html for table in tables]
    )
    print("Summarize images...")
    image_summaries = summarize_images(images_b64)

    create_docs(
        texts,
        tables,
        images_b64,
        text_summaries,
        table_summaries,
        image_summaries,
        retriever,
        id_key,
    )
