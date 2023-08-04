const aws = require('aws-sdk');

exports.handler = async (event, context) => {
    console.log('## ENVIRONMENT VARIABLES: ' + JSON.stringify(process.env));
    console.log('## EVENT: ' + JSON.stringify(event));
        
    const response = {
        statusCode: 200,
        body: "message"
    };
    return response;
};