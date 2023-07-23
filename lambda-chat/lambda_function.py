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

module_path = "."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
configTableName = os.environ.get('configTableName')
endpoint_url = os.environ.get('endpoint_url')
bedrock_region = os.environ.get('bedrock_region')
modelId = os.environ.get('model_id')
print('model_id: ', modelId)

def save_configuration(userId, modelId):
    item = {
        'user-id': {'S':userId},
        'model-id': {'S':modelId}
    }

    client = boto3.client('dynamodb')
    try:
        resp =  client.put_item(TableName=configTableName, Item=item)
        print('resp, ', resp)
    except: 
        raise Exception ("Not able to write into dynamodb")            

def load_configuration(userId):
    print('configTableName: ', configTableName)
    print('userId: ', userId)

    client = boto3.client('dynamodb')    
    try:
        key = {
            'user-id': {'S':userId}
        }

        resp = client.get_item(TableName=configTableName, Key=key)
        print('model-id: ', resp['Item']['model-id']['S'])

        return resp['Item']['model-id']['S']
    except: 
        # raise Exception ("Not able to load from dynamodb")                
        print('No record of configuration!')
        modelId = os.environ.get('model_id')
        save_configuration(userId, modelId)

        return modelId

# Bedrock Contiguration
bedrock_region = bedrock_region
bedrock_config = {
    "region_name":bedrock_region,
    "endpoint_url":endpoint_url
}
    
# supported llm list from bedrock
boto3_bedrock = bedrock.get_bedrock_client(
    region=bedrock_config["region_name"],
    url_override=bedrock_config["endpoint_url"])
    
modelInfo = boto3_bedrock.list_foundation_models()    
print('models: ', modelInfo)

llm = Bedrock(model_id=modelId, client=boto3_bedrock)

def get_summary(file_type, s3_file_name):
    summary = ''
    
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

        from langchain.document_loaders import CSVLoader
        contents = CSVLoader(reader)
    
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
        
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

    if summary == '':  # error notification
        summary = 'Fail to summarize the document. Try agan...'
        return summary
    else:
        # return summary[1:len(summary)-1]   
        return summary
    
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

    global modelId, llm
    
    modelId = load_configuration(userId)
    if(modelId==""): 
        modelId = os.environ.get('model_id')
        save_configuration(userId, modelId)

    start = int(time.time())    

    msg = ""
    if type == 'text' and body[:11] == 'list models':
        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)
    
    elif type == 'text' and body[:20] == 'change the model to ':
        new_model = body.rsplit('to ', 1)[-1]
        print(f"new model: {new_model}, current model: {modelId}")

        if modelId == new_model:
            msg = "No change! The new model is the same as the current model."
        else:        
            lists = modelInfo['modelSummaries']
            isChanged = False
            for model in lists:
                if model['modelId'] == new_model:
                    print(f"new modelId: {new_model}")
                    modelId = new_model
                    llm = Bedrock(model_id=modelId, client=boto3_bedrock)
                    isChanged = True
                    save_configuration(userId, modelId)

            if isChanged:
                msg = f"The model is changed to {modelId}"
            else:
                msg = f"{modelId} is not in lists."
        print('msg: ', msg)

    else:             
        if type == 'text':
            text = body
            msg = llm(text)
            
        elif type == 'document':
            object = body
        
            file_type = object[object.rfind('.')+1:len(object)]
            print('file_type: ', file_type)
            
            msg = get_summary(file_type, object)
                
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