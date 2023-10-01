# Chat Memory를 활용하는 방법

## Prefix 활용하여 chat memory를 이용하기

ConversationBufferMemory를 아래와 같이 정의합니다. 

```python
chat_memory = ConversationBufferMemory(
  memory_key="chat_history",
  return_messages=True,
  input_key="question",
  output_key='answer',
  human_prefix='Human',
  ai_prefix='Assistant')
```

dialog를 추가할 경우에는 아래와 같이 보기 편하게 answer에서 개행문자를 지우고 save_context을 이용하여 저장합니다.

```python
storedMsg = str(msg).replace("\n"," ") 
chat_memory.save_context({"input": text}, {"output": storedMsg})
```

chat history를 로그로 볼때는  아래와 같이 합니다.

```python
chats = chat_memory.load_memory_variables({})
chat_history_all = chats['history']
print('chat_history_all: ', chat_history_all)
```

## ConversationalRetrievalChain을 위한 chat memory 정의하기

아래와 같이 memory chain을 정의합니다.

```python
memory_chain = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
```

새로운 diaglog는 아래와 같이 추가할 수 있습니다.

```python
memory_chain.chat_memory.add_user_message(text) 
memory_chain.chat_memory.add_ai_message(msg)    
```

memory에서 chat history를 추출할 때는 아래와 같이 합니다.

```python
from langchain.schema import BaseMessage
_ROLE_MAP = {"human": "\n\nHuman: ", "ai": "\n\nAssistant: "}

chat_history = extract_chat_history_from_memory(memory_chain)

def extract_chat_history_from_memory(memory_chain):
    chat_history = []
    chats = memory_chain.load_memory_variables({})    
    for dialogue_turn in chats['chat_history']:
        role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
        chat_history.append(f"{role_prefix[2:]}{dialogue_turn.content}")

    return chat_history
```

