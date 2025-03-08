RagInstructions = """-- RAGs (retrievers) --
Use RAGs to retrieve or store data using vector-based databases.

Request information about a specific rag, or search for available rags:
START>>>
{
  "calls": [
    {
      "type": "rags", # Type can be any of rag, rags, retriever or retrievers
      "method": "request",
      "instance_name": "<exact_rag_name> or <search query>", # E.g “code_base“ or “Find function doc for X“
      "action": "",
      "parameters": {}
    }
  ]
}
<<<END

Once the correct rag is identified, execute it using one of the below:
START>>>
{
  "calls": [
    {
      "type": "rag", # Type can be any of rag, rags, retriever or retrievers
      "method": "execute",
      "instance_name": "<exact_rag_name>", # E.g “code_base“
      "action": "<action_name>",
      "parameters": {params_dictionary}
    }
  ]
}
<<<END
"""
