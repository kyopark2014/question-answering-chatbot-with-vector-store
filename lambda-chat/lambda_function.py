import json
import boto3
import os
import time
import datetime
from io import BytesIO
import PyPDF2
import csv
import sys

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.llms.bedrock import Bedrock
from langchain.chains.question_answering import load_qa_chain

from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import CSVLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
configTableName = os.environ.get('configTableName')
endpoint_url = os.environ.get('endpoint_url')
opensearch_url = os.environ.get('opensearch_url')
bedrock_region = os.environ.get('bedrock_region')
rag_type = os.environ.get('rag_type')
enableConversationMode = os.environ.get('enableConversationMode', 'enabled')
print('enableConversationMode: ', enableConversationMode)
enableReference = os.environ.get('enableReference', 'false')
enableRAG = os.environ.get('enableRAG', 'true')

# opensearch authorization - id/passwd
opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
# opensearch authorization - aws auth
# from requests_aws4auth import AWS4Auth
# credentials = boto3.Session().get_credentials()
# awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

modelId = os.environ.get('model_id')
print('model_id: ', modelId)
isReady = False   
accessType = os.environ.get('accessType')

# load documents from s3
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
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
    
    #print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    texts = text_splitter.split_text(new_contents) 
    #print('texts[0]: ', texts[0])
            
    return texts
              
def get_answer_using_query(query, vectorstore, rag_type):
    wrapper_store = VectorStoreIndexWrapper(vectorstore=vectorstore)        
    
    answer = wrapper_store.query(question=query, llm=llm)    
    print('answer: ', answer)

    return answer

def summerize_text(text):
    docs = [
        Document(
            page_content=text
        )
    ]
    prompt_template = """Write a concise summary of the following:

    {text}
                
    CONCISE SUMMARY """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    summary = chain.run(docs)
    print('summarized text: ', summary)

    return summary

#def get_chat_history(inputs):
#    inputs = [i.content for i in inputs]
#    return  '\n'.join(inputs)

chat_history = []
def get_answer_using_template_with_history(query, vectorstore, chat_history):  
    prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Assistant:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    #CONDENSE_QUESTION_TEMPLATE = """Given the following conversation and a follow up question, #rephrase the follow up question to be a standalone question.
    #Chat History:
    #{chat_history}
    #Follow Up Input: {question}
    #Standalone question:"""

    # Condense Prompt
    condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)
   
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),         
        condense_question_prompt=CONDENSE_QUESTION_PROMPT, # chat history and new question
        chain_type='stuff', # 'refine'
        verbose=False, # for logging to stdout
        #condense_question_llm
        #combine_docs_chain_kwargs={"prompt": query}  #  load_qa_chain

        rephrase_question=True,  # to pass the new generated question to the combine_docs_chain
        
        memory=memory,
        #qa_prompt=CONDENSE_QUESTION_TEMPLATE,
        #output_key='answer', 
        #max_tokens_limit=300,
        #chain_type_kwargs={"prompt": PROMPT} <-- (x)
        
        return_source_documents=True, # retrieved source
        return_generated_question=False, # generated question
        
        #get_chat_history=get_chat_history,
        get_chat_history=lambda h:h,
        
    )
    qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(prompt_template) # to combine any retrieved documents.

    
    result = qa({"question": query, "chat_history": chat_history})
    
    print('result: ', result)

    chats = memory.load_memory_variables({})
    print('chats: ', chats['chat_history'])

    
    chat_history.append([(query, result["answer"])])
    print('chat_history: ', chat_history)

    source_documents = result['source_documents']
    print('source_documents: ', source_documents)

    if len(source_documents)>=1 and enableReference == 'true':
        reference = get_reference(source_documents)
        print('reference: ', reference)

        return result['answer']+reference
    else:
        return result['answer']


def get_answer_using_template(query, vectorstore, rag_type):        
    #summarized_query = summerize_text(query)        
    #    if rag_type == 'faiss':
    #        query_embedding = vectorstore.embedding_function(summarized_query)
    #        relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
    #    elif rag_type == 'opensearch':
    #        relevant_documents = vectorstore.similarity_search(summarized_query)
    
    if rag_type == 'faiss':
        query_embedding = vectorstore.embedding_function(query)
        relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
    elif rag_type == 'opensearch':
        relevant_documents = vectorstore.similarity_search(query)

    print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    print('----')
    for i, rel_doc in enumerate(relevant_documents):
        print(f'## Document {i+1}: {rel_doc.page_content}.......')
        print('---')
    
    print('length of relevant_documents: ', len(relevant_documents))
    
    prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Assistant:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = qa({"query": query})
    print('result: ', result)
    source_documents = result['source_documents']
    print('source_documents: ', source_documents)

    if len(relevant_documents)>=1:
        reference = get_reference(source_documents)
        #print('reference: ', reference)

        return result['result']+reference
    else:
        return result['result']

def get_reference(docs):
    reference = "\n\nFrom\n"
    for doc in docs:
        name = doc.metadata['name']
        page = doc.metadata['page']
    
        reference = reference + (str(page)+'page in '+name+'\n')
    return reference

# Bedrock Contiguration
bedrock_region = bedrock_region
bedrock_config = {
    "region_name":bedrock_region,
    "endpoint_url":endpoint_url
}
    
# supported llm list from bedrock
if accessType=='aws':  # internal user of aws
    boto3_bedrock = boto3.client(
        service_name='bedrock',
        region_name=bedrock_config["region_name"],
        endpoint_url=bedrock_config["endpoint_url"],
    )
else: # preview user
    boto3_bedrock = boto3.client(
        service_name='bedrock',
        region_name=bedrock_config["region_name"],
    )

modelInfo = boto3_bedrock.list_foundation_models()    
print('models: ', modelInfo)

def get_parameter(modelId):
    if modelId == 'amazon.titan-tg1-large': 
        return {
            "maxTokenCount":1024,
            "stopSequences":[],
            "temperature":0,
            "topP":0.9
        }
    elif modelId == 'anthropic.claude-v1':
        return {
            "max_tokens_to_sample":1024,
        }
parameters = get_parameter(modelId)

llm = Bedrock(model_id=modelId, client=boto3_bedrock, model_kwargs=parameters)

# embedding
bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)

# conversation retrival chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question", output_key='answer')


#memory = ConversationBufferMemory()
#from langchain.chains import ConversationChain
#conversation = ConversationChain(
#    llm=llm, verbose=True, memory=memory
#)

#from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
#print("CONDENSE_QUESTION_PROMPT: ", CONDENSE_QUESTION_PROMPT.template)

def lambda_handler(event, context):
    print(event)
    userId  = event['user-id']
    print('userId: ', userId)
    requestId  = event['request-id']
    print('requestId: ', requestId)
    type  = event['type']
    print('type: ', type)
    body = event['body']
    print('body: ', body)

    global modelId, llm, vectorstore, isReady, enableConversationMode, enableReference, enableRAG, chat_history
    
    if rag_type == 'opensearch':
        vectorstore = OpenSearchVectorSearch(
            # index_name = "rag-index-*", // all
            index_name = 'rag-index-'+userId+'-*',
            is_aoss = False,
            #engine="faiss",  # default: nmslib
            embedding_function = bedrock_embeddings,
            opensearch_url=opensearch_url,
            http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
        )
    elif rag_type == 'faiss':
        print('isReady = ', isReady)
    
    start = int(time.time())    

    msg = ""
    if type == 'text' and body[:11] == 'list models':
        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)
    
    else:             
        if type == 'text':
            text = body

             # debugging
            if text == 'enableReference':
                enableReference = 'true'
                msg  = "Referece is enabled"
            elif text == 'disableReference':
                enableReference = 'false'
                msg  = "Reference is disabled"
            elif text == 'enableConversationMode':
                enableConversationMode = 'true'
                msg  = "Conversation mode is enabled"
            elif text == 'disableConversationMode':
                enableConversationMode = 'false'
                msg  = "Conversation mode is disabled"
            elif text == 'enableRAG':
                enableRAG = 'true'
                msg  = "RAG is enabled"
            elif text == 'disableRAG':
                enableRAG = 'false'
                msg  = "RAG is disabled"
            else:

                if rag_type == 'faiss' and isReady == False: 
                    msg = llm(text)
                else: 
                    querySize = len(text)
                    textCount = len(text.split())
                    print(f"query size: {querySize}, workds: {textCount}")

                    if querySize<1800 and enableRAG=='true': # max 1985
                        if enableConversationMode == 'true':
                            msg = get_answer_using_template_with_history(text, vectorstore, chat_history)
                        else:
                            msg = get_answer_using_template(text, vectorstore, rag_type)
                    else:
                        msg = llm(text)
                        #msg = conversation.predict(input=text)
                #print('msg: ', msg)
            
        elif type == 'document':
            object = body
        
            file_type = object[object.rfind('.')+1:len(object)]
            print('file_type: ', file_type)
            
            # load documents where text, pdf, csv are supported
            texts = load_document(file_type, object)

            docs = []
            for i in range(len(texts)):
                docs.append(
                    Document(
                        page_content=texts[i],
                        metadata={
                            'name': object,
                            'page':i+1
                        }
                    )
                )        
            print('docs[0]: ', docs[0])    
            print('docs size: ', len(docs))
                        
            if rag_type == 'faiss':
                if isReady == False:   
                    vectorstore = FAISS.from_documents( # create vectorstore from a document
                        docs,  # documents
                        bedrock_embeddings  # embeddings
                    )
                    isReady = True
                else:
                    vectorstore.add_documents(docs)
                    print('vector store size: ', len(vectorstore.docstore._dict))

            elif rag_type == 'opensearch':    
                new_vectorstore = OpenSearchVectorSearch(
                    index_name="rag-index-"+userId+'-'+requestId,
                    is_aoss = False,
                    #engine="faiss",  # default: nmslib
                    embedding_function = bedrock_embeddings,
                    opensearch_url = opensearch_url,
                    http_auth=(opensearch_account, opensearch_passwd),
                )
                new_vectorstore.add_documents(docs)     

                #vectorstore = OpenSearchVectorSearch.from_documents(
                #    docs, 
                #    bedrock_embeddings, 
                #    opensearch_url=opensearch_url,
                #    http_auth=(opensearch_account, opensearch_passwd),
                #)

            # summerization to show the document
            docs = [
                Document(
                    page_content=t
                ) for t in texts[:3]
            ]
            prompt_template = """Write a concise summary of the following:

            {text}
                
            CONCISE SUMMARY """

            PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
            summary = chain.run(docs)
            print('summary: ', summary)

            msg = summary
            # summerization
            #query = "summerize the documents"
            #msg = get_answer_using_query(query, vectorstore, rag_type)
            #print('msg1: ', msg)

            #msg = get_answer_using_template(query, vectorstore, rag_type)
            #print('msg2: ', msg)
                
        elapsed_time = int(time.time()) - start
        print("total run time(sec): ", elapsed_time)

        print('msg: ', msg)

        item = {
            'user-id': {'S':userId},
            'request-id': {'S':requestId},
            'type': {'S':type},
            'body': {'S':body},
            'msg': {'S':msg}
        }

        client = boto3.client('dynamodb')
        try:
            resp =  client.put_item(TableName=callLogTableName, Item=item)
        except: 
            raise Exception ("Not able to write into dynamodb")
        
        print('resp, ', resp)

    return {
        'statusCode': 200,
        'msg': msg,
    }