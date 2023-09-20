const myForm = document.querySelector('#my-form');
const userInput = document.querySelector('#userId');
const phonenumberInput = document.querySelector('#phonenumber');
const msg = document.querySelector('.msg');

myForm.addEventListener('submit', onSubmit);

// load userId 
userId.value = localStorage.getItem('userId');

var userId = localStorage.getItem('userId'); // set userID if exists 
if(userId != '')    {
    userInput.value = userId;
}

function onSubmit(e) {
    e.preventDefault();

    console.log(userInput.value);
    
    localStorage.setItem('userId',userInput.value);

    console.log('Save Profile> userId:', userInput.value)    

    window.location.href = "chat.html";
}

