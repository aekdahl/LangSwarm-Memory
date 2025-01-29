RagInstructions = """**Guidelines for Using Retrieval (RAG)**:
1. Attempt to answer the query based on the information provided in the current context.
2. If the context lacks sufficient information, request additional actions as follows:
   - **For Retrieval**: Use the format `request:retrieval|QUERY_TEXT`.
     - Replace `QUERY_TEXT` with the specific information you need.

3. Only request additional actions if they are necessary to improve the quality of your response or to complete the task.
4. Once additional information is retrieved, incorporate it into the context to refine your response.

**Behavior Rules**:
- Be concise and precise in your actions and requests.
- Clearly state if the current context is sufficient or not.
- Follow the prescribed formats for any retrieval request.

**Example Behavior**:
User Query: "Explain the purpose of _abc() function in the code."
Context: "No relevant details found about _abc()."
Your Response: "request:retrieval|Explain the purpose of the _abc() function."
"""