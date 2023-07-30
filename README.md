# Vector Store를 이용한 Question/Answering Chatbot 

여기서는 [Amazon Bedrock](https://aws.amazon.com/ko/bedrock/)의 대규모 언어 모델(Large Language Models)을 이용하여 질문/답변(Question/Answering)을 수행하는 chatbot을 vector store를 이용하여 구현합니다. 대량의 데이터로 사전학습(pretrained)한 대규모 언어 모델(LLM)은 학습되지 않은 질문에 대해서도 가장 가까운 답변을 맥락(context)에 맞게 찾아 답변할 수 있습니다. 이는 기존의 Role 방식보다 훨씬 정답에 가까운 답변을 제공하지만, 때로는 매우 그럴듯한 잘못된 답변(hallucination)을 할 수 있습니다. 이런 경우에 파인 튜닝(fine tuning)을 통해 정확도를 높일 수 있으나, 계속적으로 추가되는 데이터를 매번 파인 튜닝으로 처리할 수 없습니다. 따라서, [RAG(Retrieval-Augmented Generation)](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html)와 같이 기본 모델의 파라미터(weight)을 바꾸지 않고, 지식 데이터베이스(knowledge Database)에서 얻어진 외부 지식을 이용하여 정확도를 개선하는 방법을 활용할 수 있습니다. RAG는 prompt engineering 기술 중의 하나로서 vector store를 지식 데이터베이스로 이용하고 있습니다. 

Vector store는 이미지, 문서(text document), 오디오와 같은 구조화 되지 않은 컨텐츠(unstructured content)를 저장하고 검색할 수 있습니다. 특히 대규모 언어 모델(LLM)의 경우에 embedding을 이용하여 텍스트들의 연관성(sementic meaning)을 벡터(vector)로 표현할 수 있으므로, 연관성 검색(sementic search)을 통해 질문에 가장 가까운 답변을 찾을 수 있습니다. 여기서는 대표적인 In-memory vector store인 [Faiss](https://github.com/facebookresearch/faiss/wiki/Getting-started)와 persistent store이면서 대용량 병렬처리가 가능한 [Amazon OpenSearch](https://medium.com/@pandey.vikesh/rag-ing-success-guide-to-choose-the-right-components-for-your-rag-solution-on-aws-223b9d4c7280)를 이용하여 문서의 내용을 분석하고 연관성 검색(sementic search) 기능을 활용합니다. 이를 통해, 대규모 언어 모델로 질문(question)을 보내면, vector store에서 가장 유사한 문서를 찾아여 답변(answering)에 활용할 수 있습니다. 

전체적인 Architecture는 아래와 같습니다. 사용자가 업로드한 문서는 [Amazon S3](https://aws.amazon.com/ko/s3/)에 저장된 후에, embedding을 통해 vectore store에 저장됩니다. 이후 사용자가 질문을 하면 vector store를 통해 질문에 가장 가까운 문장들을 받아오고 이를 기반으로 prompt를 생성하여 대규모 언어 모델(LLM)에 질문을 요청하게 됩니다. 만약 vector store에서 질문에 가까운 문장이 없다면 대규모 언어 모델(LLM)으로 질문을 전달합니다. 대용량 파일을 S3에 업로드 할 수 있도록 [presigned url](https://docs.aws.amazon.com/ko_kr/AmazonS3/latest/userguide/PresignedUrlUploadObject.html)을 이용하였고, 질문과 답변을 수행한 call log는 [Amazon DynamoDB](https://aws.amazon.com/ko/dynamodb/)에 저장되어 이후 데이터 수집 및 분석에 사용됩니다. 여기서 대용량 언어 모델로 [Bedrock Titan](https://aws.amazon.com/ko/bedrock/titan/)을 이용합니다. Titan 모델은 Amazon에 의해서 대용량 데이터셋으로 사전훈련 되었고 강력하고 범용적인 목적을 위해 사용될 수 있습니다. 또한 LangChain을 활용하여 Application을 개발하였고, chatbot을 제공하는 인프라는 AWS CDK를 통해 배포합니다. 

<img src="https://github.com/kyopark2014/question-answering-chatbot-with-vector-store/assets/52392004/1c1ea130-43de-497e-9823-c5cf7c875f98" width="800">


문서파일을 업로드하여 vector store에 저장하는 과정은 아래와 같습니다.

1) 사용자가 파일 업로드를 요청합니다. 이때 사용하는 Upload API는 [lambda (upload)](.lambda-upload/index.js)는 S3 presigned url을 생성하여 전달합니다.
2) 이후 presigned url로 문서를 업로드 하면 S3에 Object로 저장됩니다.
3) Chat API에서 request type을 'document'로 지정하면 [lambda (chat)](.lambda-chat/index.js)은 S3에서 object를 로드하여 텍스트를 추출합니다.
4) Embeding을 통해 단어들을 vector화 합니다.
5) Vector store에 저장합니다. 이때 RAG의 type이 "faiss"이면 in-memory store인 Faiss로 저장하고, "opensearch"이면 Amazon OpenSearch로 저장합니다.

채팅 창에서 텍스트 입력(Prompt)를 통해 RAG를 활용하는 과정은 아래와 같습니다.
1) 사용자가 채팅창에서 질문(Question)을 입력합니다.
2) 이것은 '/chat' API를 이용하여 [lambda (chat)](.lambda-chat/index.js)에 전달됩니다.
3) lambda(chat)은 질문을 Embedding후에 vector store에 관련된 문장이 있는지 확인합니다.
4) Vector store가 관련된 문장을 전달하면 prompt template를 이용하여 LLM에 질문을 전달합니다. 이후 답변을 받으면 사용자에게 결과를 전달합니다.


## 주요 구성

### Bedrock을 LangChain으로 연결하기

현재(2023년 7월) Bedrock은 Preview 상태이므로, Bedrock 사용을 위해서는 AWS를 통해 사용권한을 획득하여야 합니다. 사용 권한 받으면 특정 region의 endpoint를 사용할 수 있도록 허용하는데, 아래와 같이 bedrock client에서 관련 정보를 설정할 수 있습니다. 이후 [Bedrock](https://python.langchain.com/docs/integrations/providers/bedrock)을 import하여 LangChain으로 application을 개발할 수 있습니다. 

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

[BedrockEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/bedrock)을 이용하여 Embedding을 합니다.

```python
from langchain.embeddings import BedrockEmbeddings
bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)
```

### 문서 읽어오기

[Client](https://github.com/kyopark2014/question-answering-chatbot-with-vector-store/blob/main/html/chat.js)에선 Upload API로 아래와 같이 업로드할 파일명과 Content-Type을 전달합니다.

```java
{
    "filename":"gen-ai-wiki.pdf",
    "contentType":"application/pdf"
}
```

[Lambda-upload](./lambda-upload/index.js)에서는 용량이 큰 문서 파일도 S3에 업로드할 수 있도록 presigned url을 생성합니다. 아래와 같이 s3Params를 지정하고 [getSignedUrlPromise](https://docs.aws.amazon.com/AWSJavaScriptSDK/latest/AWS/S3.html#getSignedUrlPromise-property)을 이용하여 url 정보를 Client로 전달합니다.

```java
const URL_EXPIRATION_SECONDS = 300;
const s3Params = {
    Bucket: bucketName,
    Key: s3_prefix+'/'+filename,
    Expires: URL_EXPIRATION_SECONDS,
    ContentType: contentType,
};

const uploadURL = await s3.getSignedUrlPromise('putObject', s3Params);
```

Client에서 아래와 같은 응답을 얻으면 "UploadURL"을 추출하여 문서 파일을 업로드합니다.

```java
{
   "statusCode":200,
   "body":"{\"Bucket\":\"storage-for-qa-chatbot-with-rag\",\"Key\":\"docs/gen-ai-wiki.pdf\",\"Expires\":300,\"ContentType\":\"application/pdf\",\"UploadURL\":\"https://storage-for-qa-chatbot-with-rag.s3.ap-northeast-2.amazonaws.com/docs/gen-ai-wiki.pdf?Content-Type=application%2Fpdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAZ3KIXN5TBIBMQXTK%2F20230730%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Date=20230730T055129Z&X-Amz-Expires=300&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAYaDmFwLW5vcnRoZWFzdC0yIkcwRQIhAP8or6Pr1lDHQpTIO7cTWPsB7kpkdkOdsrd2NbllPpsuAiBlV...(중량)..78d1b62f1285e8def&X-Amz-SignedHeaders=host\"}"
}
```

파일 업로드가 끝나면, [Client](./html/chat.js)는 Chat API로 문서를 vector store에 등록하도록 아래와 같이 요청합니다. 

```java
{
   "user-id":"f642fd39-8ef7-4a77-9911-1c50608c2831",
   "request-id":"d9ab57ad-6950-412e-a492-1381eb1f2642",
   "type":"document",
   "body":"gen-ai-wiki.pdf"
}
```

[Lambda-chat](./lambda-chat/lambda_function.py)에서는 type이 "document" 이라면, S3에서 아래와 같이 파일을 로드하여 text를 분리합니다.

```python
s3r = boto3.resource("s3")
doc = s3r.Object(s3_bucket, s3_prefix + '/' + s3_file_name)

if file_type == 'pdf':
    contents = doc.get()['Body'].read()
    reader = PyPDF2.PdfReader(BytesIO(contents))

    raw_text = []
    for page in reader.pages:
        raw_text.append(page.extract_text())
    contents = '\n'.join(raw_text)    
        
elif file_type == 'txt':
    contents = doc.get()['Body'].read()
elif file_type == 'csv':
    body = doc.get()['Body'].read()
    reader = csv.reader(body)
    contents = CSVLoader(reader)
```

이후 chunk size로 분리한 후에 Document를 이용하여 문서로 만듧니다.

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


### Vector Store 

Faiss와 OpenSearch 방식의 선택은 [cdk-qa-with-rag-stack.ts](./cdk-qa-with-rag/lib/cdk-qa-with-rag-stack.ts)에서 rag_type을 "faiss" 또는 "opensearch"로 변경할 수 있습니다. 기본값은 "opensearch"입니다.

#### Faiss

[Faiss](https://github.com/facebookresearch/faiss)는 Facebook에서 오픈소스로 제공하는 In-memory vector store로서 embedding과 document들을 저장할 수 있으며, [LangChain을 지원](https://python.langchain.com/docs/integrations/vectorstores/faiss)합니다. Faiss에서는 FAISS()를 이용하여 아래와 같이 vector store를 정의합니다. 

```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents( # create vectorstore from a document
    docs, 
    bedrock_embeddings  
)
```

이후, vectorstore를 이용하여 관계된 문서를 조회합니다. 이때 Faiss는 embedding된 query를 이용하여 [similarity_search_by_vector()](https://python.langchain.com/docs/modules/data_connection/vectorstores/)로 조회합니다.

```python
relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
```


#### OpenSearch

[Amazon OpenSearch persistent store로는 vector store](https://python.langchain.com/docs/integrations/vectorstores/opensearch)를 구성할 수 있습니다. 비슷한 역할을 하는 persistent store로는 RDS Postgres with pgVector, ChromaDB, Pinecone과 Weaviate가 있습니다. 

OpenSearch를 사용을 위해서는 IAM Role에서 아래의 퍼미션을 추가합니다.

```java
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "es:*",
            "Resource": "arn:aws:es:[region]:[account-id]:domain/[domain-name]/*"
        }
    ]
}
```

또한, 이때의 OpenSearch에 대한 access policy는 아래와 같습니다.

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
      "Resource": "arn:aws:es:[region]:[account-id]:domain/[domain-name]/*"
    }
  ]
}
```

이제, 아래와 같이 [OpenSearchVectorSearch()](https://python.langchain.com/docs/integrations/vectorstores/opensearch)으로 vector store를 정의합니다. 

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

### Question/Answering

아래와 같이 vector store에 직접 Query 하는 방식과, Template를 이용하는 2가지 방법으로 Question/Answering 구현하는 것을 설명합니다.

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


## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)


### CDK를 이용한 인프라 설치
[인프라 설치](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/blob/main/deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. [CDK 구현 코드](./cdk-qa-with-rag/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다.


### 실행결과

파일을 올리면 파일의 텍스트를 기반으로 요약(Summeraztion)을 수행합니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/assets/52392004/93a8d391-905b-487b-a45e-a7c67f03528b)

이후 텍스트로 질문을 하면 아래와 같이 업로드한 문서파일을 기반으로 답변을 수행합니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/assets/52392004/4050847f-8f05-4136-a290-9818f108d1cb)

잘못된 질문을 하여도 아래와 같이 업로드한 문서에 맞추어서 답변을 할 수 있습니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/assets/52392004/4be7f868-b830-4e2f-af1d-445f905f280b)

#### Chatbot 동작 시험시 주의할점

일반적인 chatbot들은 지속적인 세션을 유지 관리하기 위해서는 websocket 등을 사용하지만, 여기서 사용한 Chatbot은 API를 테스트하기 위하여 RESTful API를 사용하고 있습니다. 따라서, LLM에서 응답이 일정시간(30초)이상 지연되는 경우에 브라우저에서 답변을 볼 수 없습니다. 따라서 긴 응답시간이 필요한 경우에 CloudWatch에서 [lambda-chat](./lambda-chat/lambda_function.py)의 로그를 확인하거나, DynamoDB에 저장된 call log를 확인합니다.


### 리소스 정리하기

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. [Cloud9 console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에 접속하여 아래와 같이 삭제를 합니다.

```java
cdk destroy
```


## 결론

AWS 서울 리전에서 Amazon Bedrock과 vector store를 이용하여 질문과 답변(Question/Answering)을 수행하는 chatbot을 구현하였습니다. Amazon Bedrock은 여러개의 대용량 언어 모델을 선택하여 사용할 수 있습니다. 여기서는 Amazon Titan을 활용하여 RAG 동작을 구현하였고, 대용량 언어 모델의 환각(hallucination)을 해결할 수 있었습니다. 또한 Chatbot 어플리케이션 개발을 위해 LangChain을 활용하였고, IaC(Infrastructure as Code)로 AWS CDK를 이용하였습니다. 대용량 언어 모델은 향후 다양한 어플리케이션에서 효과적으로 활용될것으로 기대됩니다. Amazon Bedrock을 이용하여 대용량 언어 모델을 개발하면 기존 AWS 인프라와 손쉽게 연동하고 다양한 어플리케이션을 효과적으로 개발할 수 있습니다.



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

