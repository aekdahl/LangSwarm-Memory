RagInstructions = """--- RAG INSTRUCTIONS ---
(RAG stands for Retrieval-Augmented Generation. But our RAGs can both store and retrieve data.)

Request available rag instances using:
   request:rags|QUERY_TEXT
   (Replace QUERY_TEXT with your specific question.)

Example to get appropriate retrievers:
User Query: “Explain the purpose of _abc() function.”
Your Response: “request:rags|Explain the _abc() function.”
"""