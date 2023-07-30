# Vector Store를 이용한 Question/Answering Chatbot 

여기서는 Amazon Bedrock의 LLM(Large Language Models)를 이용하여 질문/답변(Question/Answering)을 수행하는 chatbot을 vector store를 이용하여 구현합니다. 대량의 데이터로 사전학습(pretrained)한 대규모 언어 모델(LLM)은 학습되지 않은 질문에 대해서도 가장 가까운 답변을 맥락(context)에 맞게 찾아 답변할 수 있습니다. 이는 기존의 Role 방식보다 훨씬 정답에 가까운 답변을 제공하지만, 때로는 매우 그럴듯한 잘못된 답변(Hallucination)을 할 수 있습니다. 이런 경우에 파인 튜닝(fine tuning)을 통해 정확도를 높일 수 있으나, 계속적으로 추가되는 데이터를 매번 파인 튜닝으로 처리할 수 없습니다. 따라서, RAG(Retrieval-Augmented Generation)와 같이 기본 모델의 파라미터(Weight등)을 바꾸지 않고, 지식 데이터베이스(knowledge DB)에서 얻어진 외부 지식을 이용하여 정확도를 개선하는 방법을 활용할 수 있습니다. RAG는 prompt engineering 기술 중의 하나로서 vector store를 지식 데이터베이스로 이용하고 있습니다. 

Vector store는 이미지, 문서(text document), 오디오와 같은 구조화 되지 않은 컨텐츠(unstructured content)를 저장하고 검색할 수 있습니다. 특히 대규모 언어 모델(LLM)의 경우에 embedding을 이용하여 텍스트들의 연관성(sementic meaning)을 벡터(Vector)로 표현할 수 있으므로, 연관성 검색(sementic search)을 통해 질문에 가장 가까운 답변을 찾을 수 있습니다. 여기서는 대표적인 In-memory vector store인 Faiss와 persistent store이면서 대용량 병렬처리가 가능한 Amazon OpenSearch를 이용하여 문서의 내용을 분석하고 sementic search 기능을 활용합니다. 이를 통해, 대규모 언어 모델로 질문(Question)을 보내면, vector store에서 가장 유사한 문서를 찾아여 답변(answering)에 사용할 수 있습니다. 이렇게 vector store를 사용하면 대규모 언어 모델의 token 사이즈를 넘어서는 긴 문장을 활용하여 질문/답변(Question/Answering)과 같은 Task를 수행할 수 있으며 환각(hallucination) 영향을 줄일 수 있습니다.

전체적인 Architecture는 아래와 같습니다. 사용자가 업로드한 문서는 Amazon S3에 저장된 후에, embedding을 통해 vectore store에 저장됩니다. 이후 사용자가 질문을 하면 vector store를 통해 질문에 가장 가까운 문장들을 받아오고 이를 기반으로 prompt를 생성하여 LLM에 질문을 요청하게 됩니다. 만약 vector store에서 질문에 가까운 문장이 없다면 LLM으로 질문을 전달합니다. 대용량 파일을 S3에 업로드 할 수 있도록 presigned url을 이용하였고, 질문과 답변을 수행한 call log는 DynamoDB에 저장되어 이후 데이터 수집 및 분석에 사용됩니다. 여기서 LLM은 Bedrock을 이용하여 LangChain 형식의 API를 통해 구현하였고, chatbot을 제공하는 인프라는 AWS CDK를 통해 배포합니다. 

<img src="https://github.com/kyopark2014/question-answering-chatbot-with-vector-store/assets/52392004/1c1ea130-43de-497e-9823-c5cf7c875f98" width="800">


문서파일을 업로드하여 vector store에 저장하는 과정은 아래와 같습니다.

1) 사용자가 파일 업로드를 요청합니다. 이때 사용하는 '/upload' API는 [lambda (upload)](.lambda-upload/index.js)는 S3 signed url을 생성하여 전달합니다.
2) 이후 signed url로 문서를 업로드 하면 S3에 Object로 저장됩니다.
3) '/chat' API에서 request type을 'document'로 지정하면 [lambda (chat)](.lambda-chat/index.js)은 S3에서 object를 로드하여 텍스트를 추출합니다.
4) Embeding을 통해 단어들을 vector화 합니다.
5) Vector store에 저장합니다. 이때 RAG의 type이 "faiss"이면 in-memory로 저장하고, "opensearch"이면 Amazon OpenSearch에 저장합니다.

채팅 창에서 텍스트 입력(Prompt)를 통해 RAG를 활용하는 과정은 아래와 같습니다.
1) 사용자가 채팅창에서 질문(Question)을 입력합니다.
2) 이것은 '/chat' API를 이용하여 [lambda (chat)](.lambda-chat/index.js)에 전달됩니다.
3) lambda(chat)은 질문을 Embedding후에 vector store에 관련된 문장이 있는지 확인합니다.
4) Vector store가 관련된 문장을 전달하면 prompt template를 이용하여 LLM에 질문을 전달합니다. 이후 답변을 받으면 사용자에게 결과를 전달합니다.


## 주요 구성

### Bedrock을 LangChain으로 연결하기

Bedrock 접속을 위해 필요한 region name과 endpoint url을 지정하고, LangChain을 사용할 수 있도록 연결하여 줍니다. Bedrock preview에서는 Dev/Prod 버전에 따라 endpoint를 달리하는데, Prod 버전을 사용하고자 할 경우에는 endpoint에 대한 부분을 삭제하거나 주석처리합니다.

```python
from langchain.llms.bedrock import Bedrock

bedrock_region = "us-west-2" 
bedrock_config = {
    "region_name":bedrock_region,
    "endpoint_url":"https://prod.us-west-2.frontend.bedrock.aws.dev"
}
    
boto3_bedrock = bedrock.get_bedrock_client(
    region=bedrock_config["region_name"],
    url_override=bedrock_config["endpoint_url"])

modelId = 'amazon.titan-tg1-large'  # anthropic.claude-v1
llm = Bedrock(model_id=modelId, client=boto3_bedrock)    
```


### Embedding

BedrockEmbeddings을 이용하여 Embedding을 합니다.

```python
from langchain.embeddings import BedrockEmbeddings
bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)
```

### Vector Store 

Faiss와 OpenSearch 방식의 선택은 [cdk-qa-with-rag-stack.ts](./cdk-qa-with-rag/lib/cdk-qa-with-rag-stack.ts)에서 rag_type을 "faiss" 또는 "opensearch"로 변경할 수 있습니다. 기본값은 "opensearch"입니다.

#### Faiss

[Faiss](https://github.com/facebookresearch/faiss)는 Facebook에서 오픈소스로 제공하는 In-memory vector store로서 embedding과 document들을 저장할 수 있으며, LangChain을 지원합니다. 비슷한 역할을 하는 persistent store로는 Amazon OpenSearch, RDS Postgres with pgVector, ChromaDB, Pinecone과 Weaviate가 있습니다. 

faiss.write_index(), faiss.read_index()을 이용해서 local에서 index를 저장하고 읽어올수 있습니다. 그러나 S3에서 직접 로드는 현재 제공하고 있지 않습니다. EFS에서 저장후 S3에 업로드 하는 방식은 레퍼런스가 있습니다.

[Faiss-LangChain](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/faiss)와 같이 save_local(), load_local()을 사용할 수 있고, merge_from()으로 2개의 vector store를 저장할 수 있습니다.

Faiss에서는 FAISS()를 이용하여 아래와 같이 vector store를 정의합니다. 

```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents( # create vectorstore from a document
    docs, 
    bedrock_embeddings  
)
```

vectorstore를 이용하여 관계된 문서를 조회합니다. 이때 Faiss는 embedding된 query를 이용하여 similarity_search_by_vector()로 조회합니다.

```python
relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
```

#### OpenSearch

OpenSearch를 사용을 위해 IAM Role에서 아래의 퍼미션을 추가합니다.

```java
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "es:*",
            "Resource": "arn:aws:es:ap-northeast-2:[account-id]:domain/[domain-name]/*"
        }
    ]
}
```

또한, OpenSearch의 access policy는 아래와 같습니다.

```java
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "*"
      },
      "Action": "es:*",
      "Resource": "arn:aws:es:ap-northeast-2:[account-id]:domain/[domain-name]/*"
    }
  ]
}
```

이제, 아래와 같이 OpenSearchVectorSearch()으로 vector store를 정의합니다. 

```python
from langchain.vectorstores import OpenSearchVectorSearch

vectorstore = OpenSearchVectorSearch.from_documents(
    docs,
    bedrock_embeddings,
    opensearch_url = endpoint_url,
    http_auth = ("admin", "password"),
)
```

아래와 같이 vectorstore로 부터 관련된 문서를 조회할 수 있습니다. 이때 OpenSearch는 query를 이용하여 similarity_search()로 조회합니다.

```python
relevant_documents = vectorstore.similarity_search(query)
```

### 문서 등록

문서를 업로드하면 FAISS를 이용하여 vector store에 저장합니다. 파일을 여러번 업로드할 경우에는 기존 vector store에 추가합니다. 

```python
docs = load_document(file_type, object)

vectorstore_new = FAISS.from_documents(
    docs,
    bedrock_embeddings,
)

vectorstore.merge_from(vectorstore_new)
```

업로드한 문서 파일에 대한 요약(Summerization)을 제공하여 사용자의 파일에 대한 이해를 돕습니다.

```python
query = "summerize the documents"

msg = get_answer(query, vectorstore_new)
print('msg2: ', msg)
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

### Question/Answering

아래와 같이 vector store에 직접 Query 하는 방식과, Template를 이용하는 2가지 방법으로 Question/Answering을 수행합니다.

#### Vector Store에서 query를 이용하는 방법

embedding한 query를 가지고 vectorstore에서 검색한 후에 vectorstore의 query()를 이용하여 답변을 얻습니다.

```python
wrapper_store = VectorStoreIndexWrapper(vectorstore = vectorstore)
query_embedding = vectorstore.embedding_function(query)

relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
answer = wrapper_store.query(question = query, llm = llm)
```

#### Template를 이용하는 방법

일반적으로 vectorstore에서 query를 이용하는 방법보다 나은 결과를 얻습니다.

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

query_embedding = vectorstore.embedding_function(query)
relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)

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
    retriever = vectorstore.as_retriever(
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
[인프라 설치](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/blob/main/deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. [CDK 구현 코드](./cdk-qa-with-rag/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다.


## 실행결과

파일을 올리면 파일의 텍스트를 기반으로 요약(Summeraztion)을 수행합니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/assets/52392004/93a8d391-905b-487b-a45e-a7c67f03528b)

이후 텍스트로 질문을 하면 아래와 같이 업로드한 문서파일을 기반으로 답변을 수행합니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/assets/52392004/4050847f-8f05-4136-a290-9818f108d1cb)

잘못된 질문을 하여도 아래와 같이 업로드한 문서에 맞추어서 답변을 할 수 있습니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/assets/52392004/4be7f868-b830-4e2f-af1d-445f905f280b)

## 브라우저에서 Chatbot 동작 시험시 주의할점

Chatbot API를 테스트하기 위해 제공하는 Web client는 일반적인 채팅 App처럼 세션 방식(websocket등)이 아니라 RESTful API를 사용합니다. 따라서 LLM에서 응답이 일정시간(30초)이상 지연되는 경우에 답변을 볼 수 없습니다. 브라우저 자체적인 timeout으로 인하여 30초 이상인 경우에 client에서 더이상 응답을 받을 수 없습니다. 이때 응답을 확인하기 위해서는 CloudWatch에서 [lambda-chat](./lambda-chat/lambda_function.py)의 로그를 확인하거나, DynamoDB에 저장된 call log를 확인합니다.


## Reference 

[Getting started - Faiss](https://github.com/facebookresearch/faiss/wiki/Getting-started)

[FAISS - LangChain](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/faiss)

[Welcome to Faiss Documentation](https://faiss.ai/)

[Adding a FAISS or Elastic Search index to a Dataset](https://huggingface.co/docs/datasets/v1.6.1/faiss_and_ea.html)

[Python faiss.write_index() Examples](https://www.programcreek.com/python/example/112290/faiss.write_index)

[OpenSearch - Langchain](https://python.langchain.com/docs/integrations/vectorstores/opensearch)

[OpenSearch - Domain](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_opensearchservice.Domain.html)

[Domain - CDK](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_opensearchservice.Domain.html)

[interface CapacityConfig - CDK](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_opensearchservice.CapacityConfig.html)

