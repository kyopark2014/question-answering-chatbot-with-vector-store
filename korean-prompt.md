# 한글 Prompt

## Basic RAG

```python
# check korean
pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
word_kor = pattern_hangul.search(str(query))
print('word_kor: ', word_kor)

if word_kor:
    prompt_template = """\n\nHuman: 아래 문맥(context)을 참조했음에도 답을 알 수 없다면, 솔직히 모른다고 말합니다.

    { context }

    Question: { question }

    Assistant: """
else:
    prompt_template = """\n\nHuman: Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor.

    { context }

    Question: { question }

    Assistant: """
```

