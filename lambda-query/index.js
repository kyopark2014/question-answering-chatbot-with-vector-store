const aws = require('aws-sdk');

const dynamo = new aws.DynamoDB.DocumentClient();
const tableName = process.env.tableName;

exports.handler = async (event, context) => {
    //console.log('## ENVIRONMENT VARIABLES: ' + JSON.stringify(process.env));
    //console.log('## EVENT: ' + JSON.stringify(event));

    const userId = event['user-id'];
    const requestId = event['request-id'];

    try {
        var params = {
            Key: {
                "user-id": {"S": userId}, 
                "request-id": {"S": requestId}
            }, 
            TableName: tableName
        };
        var result = await dynamodb.getItem(params).promise()
        console.log(JSON.stringify(result))
    } catch (error) {
        console.error(error);
    }

    const response = {
        statusCode: 200,
        msg: "message"
    };
    return response;
};