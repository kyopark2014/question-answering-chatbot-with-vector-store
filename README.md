# 질문/답변을 하는 챗봇 만들기 

여기서는 문서들을 업로드하면 Vector store에 저장후 이를 이용하여 Question/Answering을 제공하는 챗봇을 만드는것을 설명합니다.

<img src="https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/assets/52392004/95780f39-b732-4dd3-b5dc-0c32f60535ca" width="750">

## FAISS

[FAISS](https://github.com/facebookresearch/faiss)는 Facebook에서 오픈소스로 제공하는 In-memory Vector Store로서 Embedding과 Document들을 저장할 수 있으며, LangChain을 지원합니다. 비슷한 역할을 하는 Persistent Store로는 Amazon OpenSearch, RDS Postgres with pgVector, ChromaDB, Pinecone과 Weaviate가 있습니다. 


## 실습하기

### CDK를 이용한 인프라 설치
[인프라 설치](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/blob/main/deployment.md)에 따라 CDK로 인프라 설치를 진행합니다.


## Reference 

[Getting started - Faiss](https://github.com/facebookresearch/faiss/wiki/Getting-started)

