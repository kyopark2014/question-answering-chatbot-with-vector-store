const aws = require('aws-sdk');

let dynamo = new aws.DynamoDB();
const tableName = process.env.tableName;

var docClient = new aws.DynamoDB.DocumentClient({ apiVersion: '2012-08-10' });

function deleteItems(userId, requestTime) {
    let deleteParams = {
        Key: {
            "user_id": userId,
            "request_time": requestTime
        },
        TableName: tableName,
    };

    docClient.delete(deleteParams, function (err, data) {
        if (err) {
            console.log("Error", err);
        } else {
            console.log("Success", data);
        }
    });
}

let isCompleted = false;

function wait() {
    return new Promise((resolve, reject) => {
        if (!isCompleted) {
            setTimeout(() => resolve("wait..."), 1000);
        }
        else {
            setTimeout(() => resolve("done..."), 0);
        }
    });
}

exports.handler = async (event, context) => {
    //console.log('## ENVIRONMENT VARIABLES: ' + JSON.stringify(process.env));
    //console.log('## EVENT: ' + JSON.stringify(event));

    const userId = event['userId'];
    console.log('userId: ', userId);

    let queryParams = {
        TableName: tableName,
        KeyConditionExpression: "user_id = :userId",
        ExpressionAttributeValues: {
            ":userId": { 'S': userId }
        }
    };

    try {
        let result = await dynamo.query(queryParams).promise();

        console.log('result: ', JSON.stringify(result));

        for (let item of result['Items']) {
            console.log('item: ', item);
            const requestTime = item['request_time']['S'];
            console.log(`userId: ${userId}, requestTime: ${requestTime}`);

            deleteItems(userId, requestTime);
        }
        isCompleted = true;

        console.log(await wait());
        console.log(await wait());
        console.log(await wait());
        console.log(await wait());
        console.log(await wait());

        const response = {
            statusCode: 200,
            msg: "done"
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