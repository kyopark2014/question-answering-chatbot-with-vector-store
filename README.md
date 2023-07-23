# 질문/답변을 하는 챗봇 만들기 

여기서는 문서들을 업로드하면 Vector store에 저장후 이를 이용하여 Question/Answering을 제공하는 챗봇을 만드는것을 설명합니다. vector store를 사용하면 LLM의 token 사이즈를 긴문장을 활용할 수 있습니다. Faiss는 빠르게 semantic search를 할 수 있도록 도와줍니다.

사용자가 파일을 로드하면 CloudFont와 API Gateway를 거쳐서 [Lambda (upload)](./lambda-upload/index.js)가 S3에 파일을 저장합니다. 저장이 완료되면 해당 Object의 bucket과 key를 이용하여 [Lambda (chat)](./lambda-chat/lambda_function.py)이 파일을 로드하여 text를 추출합니다. text는 chunk size로 분리되어서 embedding을 통해 Faiss에 index로 저장됩니다. 사용자가 메시지를 전달하면 Faiss로 부터 가장 가까운 chunk를 3개 문장들을 가지고 Question/Answering을 수행합니다. 이후 관련된 call log는 DynamoDB에 저장됩니다. 여기서 LLM은 Bedrock을 LangChain 형식의 API를 통해 구현하였고, Chatbot을 제공하는 인프라는 AWS CDK를 통해 배포합니다. 

<img src="https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/assets/52392004/69a6fe75-1108-4fcb-a64b-807501076360" width="750">


## Faiss

[Faiss](https://github.com/facebookresearch/faiss)는 Facebook에서 오픈소스로 제공하는 In-memory vector store로서 embedding과 document들을 저장할 수 있으며, LangChain을 지원합니다. 비슷한 역할을 하는 persistent store로는 Amazon OpenSearch, RDS Postgres with pgVector, ChromaDB, Pinecone과 Weaviate가 있습니다. 

faiss.write_index(), faiss.read_index()을 이용해서 local에서 index를 저장하고 읽어올수 있습니다. 그러나 S3에서 직접 로드는 현재 제공하고 있지 않습니다. EFS에서 저장후 S3에 업로드 하는 방식은 레퍼런스가 있습니다.

[Faiss-LangChain](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/faiss)와 같이 save_local(), load_local()을 사용할 수 있고, merge_from()으로 2개의 vector store를 저장할 수 있습니다.


## 주요 구성

### 문서 업로드후 요약하기

문서를 업로드하면 FAISS를 이용하여 vector store에 저장합니다. 파일을 여러번 업로드할 경우에는 기존 vector store에 추가합니다. 

```python
docs = load_document(file_type, object)

vectorstore_faiss_new = FAISS.from_documents(
    docs,
    bedrock_embeddings,
)

vectorstore_faiss.merge_from(vectorstore_faiss_new)
print('vector store size: ', len(vectorstore_faiss.docstore._dict))

query = "summerize the documents"

msg = get_answer(query, vectorstore_faiss_new)
print('msg2: ', msg)
```

### Embedding

Embedding으로 BedrockEmbeddings을 사용합니다.

```python
from langchain.embeddings import BedrockEmbeddings
bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)
```

### 파일 읽어오기

pdf, txt, csv 파일을 S3에서 로딩하여 chunk size로 분리한 후에 Document를 이용하여 문서로 만듧니다.

```python
from langchain.docstore.document import Document

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
texts = text_splitter.split_text(new_contents)
print('texts[0]: ', texts[0])

docs = [
    Document(
        page_content = t
    ) for t in texts[: 3]
    ]
return docs
```

## Question/Aswering

### vectorstore에서 query를 이용하는 방법

embedding한 query를 가지고 vectorstore에서 검색한 후에 vectorstore의 query()를 이용하여 답변을 얻습니다.

```python
wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore = vectorstore_faiss)
query_embedding = vectorstore_faiss.embedding_function(query)

relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)
answer = wrapper_store_faiss.query(question = query, llm = llm)
```

### Question/Answer Template를 이용하는 방법

일반적으로 vectorstore에서 query를 이용하는 방법보다 나은 결과를 얻습니다.

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

query_embedding = vectorstore_faiss.embedding_function(query)
relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)

    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{ context }

Question: { question }
Assistant: """
PROMPT = PromptTemplate(
    template = prompt_template, input_variables = ["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = vectorstore_faiss.as_retriever(
        search_type = "similarity", search_kwargs = { "k": 3 }
    ),
    return_source_documents = True,
    chain_type_kwargs = { "prompt": PROMPT }
)
result = qa({ "query": query })

return result['result']
```

## 실습하기

### CDK를 이용한 인프라 설치
[인프라 설치](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/blob/main/deployment.md)에 따라 CDK로 인프라 설치를 진행합니다.


## Reference 

[Getting started - Faiss](https://github.com/facebookresearch/faiss/wiki/Getting-started)

[FAISS - LangChain](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/faiss)

[Welcome to Faiss Documentation](https://faiss.ai/)

[Adding a FAISS or Elastic Search index to a Dataset](https://huggingface.co/docs/datasets/v1.6.1/faiss_and_ea.html)

[Python faiss.write_index() Examples](https://www.programcreek.com/python/example/112290/faiss.write_index)

