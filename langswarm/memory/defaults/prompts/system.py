RagInstructions = """--- RAG INSTRUCTIONS ---
(RAG stands for Retrieval-Augmented Generation. But our RAGs can both store and retrieve data.)

Request available rag instances and instructions on how to use them:
   request:rags|QUERY_TEXT
   (Replace QUERY_TEXT with a specific rag name, or a search query.)

Example to get appropriate retrievers:
User Query: “Explain the purpose of _abc() function.”
Your Response: “request:rags|chroma_db_tool” or “request:rags|Explain the _abc() function.”
"""