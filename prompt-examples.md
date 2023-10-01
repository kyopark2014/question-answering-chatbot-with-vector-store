# Prompt Examples

```python
prompt_template = """Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{ context }

Question: { question }

Assistant: """

PROMPT = PromptTemplate(
    template = prompt_template, input_variables = ["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {
            "k": 3, 'score_threshold': 0.8
        }
    ),
    return_source_documents = True,
    chain_type_kwargs = { "prompt": PROMPT }
)
result = qa({ "query": query })
```
