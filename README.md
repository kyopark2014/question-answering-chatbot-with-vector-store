# 질문/답변을 하는 챗봇 만들기 

여기서는 문서들을 업로드하면 Vector store에 저장후 이를 이용하여 Question/Answering을 제공하는 챗봇을 만드는것을 설명합니다.

<img src="https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/assets/52392004/95780f39-b732-4dd3-b5dc-0c32f60535ca" width="750">

## FAISS

[FAISS](https://github.com/facebookresearch/faiss)는 Facebook에서 오픈소스로 제공하는 In-memory vector store로서 embedding과 document들을 저장할 수 있으며, LangChain을 지원합니다. 비슷한 역할을 하는 persistent store로는 Amazon OpenSearch, RDS Postgres with pgVector, ChromaDB, Pinecone과 Weaviate가 있습니다. 

faiss.write_index(), faiss.read_index()을 이용해서 local에서 index를 저장하고 읽어올수 있습니다. 그러나 S3에서 직접 로드는 현재 제공하고 있지 않습니다. EFS에서 저장후 S3에 업로드 하는 방식은 레퍼런스가 있습니다.

[FAISS](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/faiss)와 같이 save_local(), load_local()을 사용할 수 있고, merge_from()으로 2개의 vector store를 저장할 수 있습니다.


## 실습하기

### CDK를 이용한 인프라 설치
[인프라 설치](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/blob/main/deployment.md)에 따라 CDK로 인프라 설치를 진행합니다.


## Reference 

[Getting started - Faiss](https://github.com/facebookresearch/faiss/wiki/Getting-started)

[FAISS - LangChain](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/faiss)

[Welcome to Faiss Documentation](https://faiss.ai/)

[Adding a FAISS or Elastic Search index to a Dataset](https://huggingface.co/docs/datasets/v1.6.1/faiss_and_ea.html)

[Python faiss.write_index() Examples](https://www.programcreek.com/python/example/112290/faiss.write_index)

