# Code Backup

## ConversationalRetrievalChain
ConversationalRetrievalChain을 테스트하기 위해 만든 코드입니다. 제대로 동작안해서 수정중입니다.

```python
msg = get_answer_using_ConversationalRetrievalChain(text, vectorstore, chat_memory)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question", output_key='answer', human_prefix='Human', ai_prefix='Assistant')

def get_answer_using_ConversationalRetrievalChain(query, vectorstore, chat_memory):  
    condense_template = """Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {chat_history}
    
    Human: {question}

    Assistant:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),         
        condense_question_prompt=CONDENSE_QUESTION_PROMPT, # chat history and new question
        chain_type='stuff', # 'refine'
        verbose=False, # for logging to stdout
        rephrase_question=True,  # to pass the new generated question to the combine_docs_chain
        
        memory=memory,
        #max_tokens_limit=300,
        return_source_documents=True, # retrieved source
        return_generated_question=False, # generated question
    )

    # combine any retrieved documents.
    prompt_template = """\n\nHuman: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    
    Assistant:"""
    qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(prompt_template) 
    
    # extract chat history
    chats = chat_memory.load_memory_variables({})
    chat_history_all = chats['history']
    print('chat_history_all: ', chat_history_all)

    # use last two chunks of chat history
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=0)
    texts = text_splitter.split_text(chat_history_all) 

    pages = len(texts)
    print('pages: ', pages)

    if pages >= 2:
        chat_history = f"{texts[pages-2]} {texts[pages-1]}"
    elif pages == 1:
        chat_history = texts[0]
    else:  # 0 page
        chat_history = ""
    print('chat_history:\n ', chat_history)

    # make a question using chat history
    result = qa({"question": query, "chat_history": chat_history})    
    print('result: ', result)    
    
    # get the reference
    source_documents = result['source_documents']
    print('source_documents: ', source_documents)

    if len(source_documents)>=1 and enableReference=='true':
        reference = get_reference(source_documents)
        #print('reference: ', reference)
        return result['answer']+reference
    else:
        return result['answer']
```
