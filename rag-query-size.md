# RAG Query Size Limitation

RAG에 보내는 Query의 사이즈 제한이 있습니다.

- OpenSearch
  - 에러 메시지: ValueError: Error raised by inference endpoint: An error occurred (ValidationException) when calling the InvokeModel operation: The provided inference configurations are invalid
  - [Approximate k-NN search](https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/): The knn_vector data type supports a vector of floats that can have a dimension count of up to 16,000 for the nmslib and faiss engines, as set by the dimension mapping parameter. The maximum dimension count for the Lucene library is 1,024.
  - [KNN Search with OpenSearch and OpenAI Embeddings: An In-Depth Guide](https://blog.reactivesearch.io/knn-search-with-opensearch-and-openai-embeddings-an-in-depth-guide): OpenSearch는 limitation 없다고 함 (확인중)

- Kendra
  - 에러 메시지: ValidationException: An error occurred (ValidationException) when calling the Retrieve operation: The provided QueryText has a character count of 3630, which exceeds the limit. The character count must be less than or equal to 1000.
  - Kendra의 Size 제한관련은 Quota 조정이 있을수 있으나, 서비스팀에서 항상 허락하는것은 아닙니다.


