INDEX_NAME = "rag-qa-test"

SYSTEM_PROMPT = """You are a helpful assistant that answers strictly from the provided context.
- If the answer is not in the context, say you don't know.
- Always include citations as [source: <metadata.source or file name>, p.<page if any>].
"""

OUTPUT_PATH = "../content/"
