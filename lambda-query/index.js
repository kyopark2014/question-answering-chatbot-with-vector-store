const aws = require('aws-sdk');

var dynamo = new aws.DynamoDB();
const tableName = process.env.tableName;
const indexName = process.env.indexName;

exports.handler = async (event, context) => {
    //console.log('## ENVIRONMENT VARIABLES: ' + JSON.stringify(process.env));
    //console.log('## EVENT: ' + JSON.stringify(event));

    let requestId = event['request_id'];
    console.log('requestId: ', requestId);    
    
    let msg = "";
    let queryParams = {
        TableName: tableName,
        IndexName: indexName, 
        KeyConditionExpression: "request_id = :requestId",
        ExpressionAttributeValues: {
            ":requestId": {'S': requestId}
        }
    };
    
    try {
        let result = await dynamo.query(queryParams).promise();    
        // console.log('result: ', JSON.stringify(result));    

        if(result['Items'])
            msg = result['Items'][0]['msg']['S'];

        console.log('msg: ', msg);   
        const response = {
            statusCode: 200,
            msg: msg
        };
        return response;
    } catch (error) {
        console.log(error);
        const response = {
            statusCode: 500,
            msg: error
        };
        return response;
    }     
};