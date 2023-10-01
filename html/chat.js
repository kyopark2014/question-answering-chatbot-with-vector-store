// Documents
const title = document.querySelector('#title');
const sendBtn = document.querySelector('#sendBtn');
const message = document.querySelector('#chatInput')
const chatPanel = document.querySelector('#chatPanel');

HashMap = function() {
    this.map = new Array();
};

HashMap.prototype = {
    put: function(key, value) {
        this.map[key] = value;
    },
    get: function(key) {
        return this.map[key];
    },
    getAll: function() {
        return this.map;
    },
    clear: function() {
        return this.map;
    },
    isEmpty: function() {
        return (this.map.size()==0);
    },
    remove: function(key) {
        delete this.map[key];
    },
    getKeys: function() {
        var keys = new Array();
        for(i in this.map) {
            keys.push(i);
        }
        return keys;
    }
};

// message log list
var msglist = [];
var maxMsgItems = 200;
var msgHistory = new HashMap();
var callee = "AWS";
var index=0;

var userId = localStorage.getItem('userId'); // set userID if exists 
if(userId=="") {
    userId = uuidv4();
}
console.log('userId: ', userId);

for (i=0;i<maxMsgItems;i++) {
    msglist.push(document.getElementById('msgLog'+i));

    // add listener        
    (function(index) {
        msglist[index].addEventListener("click", function() {
            if(msglist.length < maxMsgItems) i = index;
            else i = index + maxMsgItems;

            console.log('click! index: '+index);
        })
    })(i);
}

calleeName.textContent = "Chatbot";  
calleeId.textContent = "AWS";

index = 0;

addNotifyMessage("Start chat with Amazon Bedrock");

addReceivedMessage("Welcome to Amazon Bedrock. Use the enhanced question and answering using RAG (Retrieval-Augmented Generation)");

// get history
function getAllowTime() {    
    let allowableDays = 2; // two day's history
    
    let current = new Date();
    let allowable = new Date(current.getTime() - 24*60*60*1000*allowableDays);  
    let allowTime = getDate(allowable)+' '+getTime(current);
    console.log('Current Time: ', getDate(current)+' '+getTime(current));
    console.log('Allow Time: ', allowTime);
    
    return allowTime;
}
let allowTime = getAllowTime();
getHistory(userId, allowTime);

// Listeners
message.addEventListener('keyup', function(e){
    if (e.keyCode == 13) {
        onSend(e);
    }
});

// refresh button
refreshChatWindow.addEventListener('click', function(){
    console.log('go back user input menu');
    window.location.href = "index.html";
});

// depart button
depart.addEventListener('click', function(){
    console.log('depart icon');
    
    deleteItems(userId);    
});

sendBtn.addEventListener('click', onSend);
function onSend(e) {
    e.preventDefault();
    
    if(message.value != '') {
        console.log("msg: ", message.value);

        let current = new Date();
        let datastr = getDate(current);
        let timestr = getTime(current);
        let requestTime = datastr+' '+timestr
        addSentMessage(message.value, timestr);
        sendRequest(message.value, requestTime);
    }
    else {
        console.log("msg: ", "empty!");
    }    
    message.value = "";

    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
}

function uuidv4() {
    return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
      (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}

(function() {
    window.addEventListener("focus", function() {
//        console.log("Back to front");

        if(msgHistory.get(callee))
            updateCallLogToDisplayed();
    })
})();

function getDate(current) {    
    return current.toISOString().slice(0,10);
}

function getTime(current) {
    let time_map = [current.getHours(), current.getMinutes(), current.getSeconds()].map((a)=>(a < 10 ? '0' + a : a));
    return time_map.join(':');
}

function addSentMessage(text, timestr) {
    index++;

    var length = text.length;    
    console.log('length: ', length);
    if(length < 10) {
        msglist[index].innerHTML = 
            `<div class="chat-sender20 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;   
    }
    else if(length < 14) {
        msglist[index].innerHTML = 
            `<div class="chat-sender25 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;   
    }
    else if(length < 17) {
        msglist[index].innerHTML = 
            `<div class="chat-sender30 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;
    }  
    else if(length < 21) {
        msglist[index].innerHTML = 
            `<div class="chat-sender35 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;
    }
    else if(length < 25) {
        msglist[index].innerHTML = 
            `<div class="chat-sender40 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;
    }
    else if(length < 35) {
        msglist[index].innerHTML = 
            `<div class="chat-sender50 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;
    }
    else if(length < 80) {
        msglist[index].innerHTML = 
            `<div class="chat-sender60 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;
    }  
    else if(length < 145) {
        msglist[index].innerHTML = 
            `<div class="chat-sender70 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;
    }  
    else {
        msglist[index].innerHTML = 
            `<div class="chat-sender80 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;
    }     

    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
}       

function addSentMessageForSummary(text, timestr) {  
    console.log("sent message: "+text);
    index++;

    let length = text.length;
    if(length < 100) {
        msglist[index].innerHTML = 
            `<div class="chat-sender60 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;   
    }
    else {
        msglist[index].innerHTML = 
            `<div class="chat-sender80 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;
    }   

    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
}  

function addReceivedMessage(msg) {
    // console.log("add received message: "+msg);
    sender = "Chatbot"
    index++;

    msg = msg.replaceAll("\n", "<br/>");

    var length = msg.length;
    console.log("length: ", length);

    if(length < 10) {
        msglist[index].innerHTML = `<div class="chat-receiver20 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 14) {
        msglist[index].innerHTML = `<div class="chat-receiver25 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 17) {
        msglist[index].innerHTML = `<div class="chat-receiver30 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 21) {
        msglist[index].innerHTML = `<div class="chat-receiver35 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 25) {
        msglist[index].innerHTML = `<div class="chat-receiver40 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 35) {
        msglist[index].innerHTML = `<div class="chat-receiver50 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 80) {
        msglist[index].innerHTML = `<div class="chat-receiver60 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 145) {
        msglist[index].innerHTML = `<div class="chat-receiver70 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else {
        msglist[index].innerHTML = `<div class="chat-receiver80 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
     
    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
}

function addNotifyMessage(msg) {
    index++;

    msglist[index].innerHTML =  
        `<div class="notification-text">${msg}</div>`;     
}

refreshChatWindow.addEventListener('click', function(){
    console.log('update chat window');
    // updateChatWindow(callee);
});

attachFile.addEventListener('click', function(){
    console.log('click: attachFile');

    let input = $(document.createElement('input')); 
    input.attr("type", "file");
    input.trigger('click');    
    
    $(document).ready(function() {
        input.change(function(evt) {
            var input = this;
            var url_file = $(this).val();
            var ext = url_file.substring(url_file.lastIndexOf('.') + 1).toLowerCase();
            var filename = url_file.substring(url_file.lastIndexOf('\\') + 1).toLowerCase();

            console.log('url: ' + url_file);
            console.log('filename: ' + filename);
            console.log('ext: ' + ext);

            if(ext == 'pdf') {
                contentType = 'application/pdf'           
            }
            else if(ext == 'txt') {
                contentType = 'text/plain'
            }
            else if(ext == 'csv') {
                contentType = 'text/csv'
            }

            let current = new Date();
            let datastr = getDate(current);
            let timestr = getTime(current);
            let requestTime = datastr+' '+timestr
            addSentMessageForSummary("uploading the selected file in order to summerize...", timestr);

            const uri = "upload";
            const xhr = new XMLHttpRequest();
        
            xhr.open("POST", uri, true);
            xhr.onreadystatechange = () => {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    response = JSON.parse(xhr.responseText);
                    console.log("response: " + JSON.stringify(response));
                                        
                    // upload the file
                    const body = JSON.parse(response.body);
                    console.log('body: ', body);

                    const uploadURL = body.UploadURL;                    
                    console.log("UploadURL: ", uploadURL);

                    var xmlHttp = new XMLHttpRequest();
                    xmlHttp.open("PUT", uploadURL, true);       

                    //let formData = new FormData();
                    //formData.append("attachFile" , input.files[0]);
                    //console.log('uploading file info: ', formData.get("attachFile"));

                    const blob = new Blob([input.files[0]], { type: contentType });

                    xmlHttp.onreadystatechange = function() {
                        if (xmlHttp.readyState == XMLHttpRequest.DONE && xmlHttp.status == 200 ) {
                            console.log(xmlHttp.responseText);
                                           
                            // summary for the upload file
                            sendRequestForSummary(filename, requestTime);
                        }
                        else if(xmlHttp.readyState == XMLHttpRequest.DONE && xmlHttp.status != 200) {
                            console.log('status' + xmlHttp.status);
                            alert("Try again! The request was failed.");
                        }
                    };
        
                    xmlHttp.send(blob); 
                    // xmlHttp.send(formData); 
                    console.log(xmlHttp.responseText);
                }
            };
        
            var requestObj = {
                "filename": filename,
                "contentType": contentType,
            }
            console.log("request: " + JSON.stringify(requestObj));
        
            var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});
        
            xhr.send(blob);       
        });
    });
       
    return false;
});

let isResponsed = new HashMap();
let retryNum = new HashMap();
function sendRequest(text, requestTime) {
    const uri = "chat";
    const xhr = new XMLHttpRequest();

    let requestId = uuidv4();
    isResponsed.put(requestId, false);
    retryNum.put(requestId, 12); // max 60s (5x12)

    xhr.open("POST", uri, true);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            response = JSON.parse(xhr.responseText);
            console.log("response: " + JSON.stringify(response));
            
            addReceivedMessage(response.msg)
        }
        else if(xhr.readyState ===4 && xhr.status === 504) {
            console.log("response: " + xhr.readyState + ', xhr.status: '+xhr.status);

            getResponse(requestId);
        }
        else {
            console.log("response: " + xhr.readyState + ', xhr.status: '+xhr.status);
        }
    };

    var requestObj = {
        "user_id": userId,
        "request_id": requestId,
        "request_time": requestTime,
        "type": "text",
        "body":text
    }
    console.log("request: " + JSON.stringify(requestObj));

    var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});

    xhr.send(blob);            
}

function sendRequestForSummary(object, requestTime) {
    const uri = "chat";
    const xhr = new XMLHttpRequest();

    let requestId = uuidv4();
    isResponsed.put(requestId, false);
    retryNum.put(requestId, 60); // max 300s (5x60)

    xhr.open("POST", uri, true);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            response = JSON.parse(xhr.responseText);
            console.log("response: " + JSON.stringify(response));
            
            addReceivedMessage(response.msg)
        }
        else if(xhr.readyState ===4 && xhr.status === 504) {
            console.log("response: " + xhr.readyState + ', xhr.status: '+xhr.status);

            getResponse(requestId);
        }
        else {
            console.log("response: " + xhr.readyState + ', xhr.status: '+xhr.status);
        }
    };
    
    var requestObj = {
        "user_id": userId,
        "request_id": requestId,
        "request_time": requestTime,
        "type": "document",
        "body": object
    }
    console.log("request: " + JSON.stringify(requestObj));

    var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});

    xhr.send(blob);            
}

function delay(ms = 1000) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
async function getResponse(requestId) {
    await delay(5000);
    
    let n = retryNum.get(requestId);
    if(n == 0) {
        console.log('Failed!')
        return;
    }
    else {
        console.log('Retry!');
        retryNum.put(requestId, n-1);
        sendRequestForRetry(requestId);
    }    
}

function sendRequestForRetry(requestId) {
    const uri = "query";
    const xhr = new XMLHttpRequest();

    xhr.open("POST", uri, true);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            response = JSON.parse(xhr.responseText);
            console.log("response: " + JSON.stringify(response));
                        
            if(response.msg) {
                isResponsed.put(requestId, true);
                addReceivedMessage(response.msg);        
                
                console.log('completed!');
            }            
            else {
                console.log('The request is not completed yet.');

                getResponse(requestId);
            }
        }
    };
    
    var requestObj = {
        "request_id": requestId,
    }
    console.log("request: " + JSON.stringify(requestObj));

    var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});

    xhr.send(blob);            
}

function getHistory(userId, allowTime) {
    const uri = "history";
    const xhr = new XMLHttpRequest();

    xhr.open("POST", uri, true);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            let response = JSON.parse(xhr.responseText);
            let history = JSON.parse(response['msg']);
            // console.log("history: " + JSON.stringify(history));
                        
            for(let i=0; i<history.length; i++) {
                if(history[i].type=='text') {                
                    // let timestr = history[i].request_time.substring(11, 19);
                    let timestr = history[i].request_time;
                    let msg = history[i].msg;
                    let body = history[i].body;
                    addSentMessage(body, timestr)
                    addReceivedMessage(msg);                            
                }                 
            }         
            if(history.length>=1) {
                addNotifyMessage("Welcome back to the conversation");               
                chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
            }
        }
    };
    
    var requestObj = {
        "userId": userId,
        "allowTime": allowTime
    }
    console.log("request: " + JSON.stringify(requestObj));

    var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});

    xhr.send(blob);            
}

function deleteItems(userId) {
    const uri = "delete";
    const xhr = new XMLHttpRequest();

    xhr.open("POST", uri, true);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            let response = JSON.parse(xhr.responseText);
            console.log("response: " + JSON.stringify(response));

            window.location.href = "index.html";
        }
    };
    
    var requestObj = {
        "userId": userId
    }
    console.log("request: " + JSON.stringify(requestObj));

    var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});

    xhr.send(blob);            
}
