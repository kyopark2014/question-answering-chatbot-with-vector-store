FROM amazon/aws-lambda-python:3.9

WORKDIR /var/task/lambda-chat

COPY lambda_function.py /var/task/
COPY test.py /var/task/   
COPY utils /var/task/utils
COPY . ..

RUN /var/lang/bin/python3.9 -m pip install --upgrade pip

RUN /var/lang/bin/python3 -m pip install PyPDF2
RUN /var/lang/bin/python3 -m pip install langchain
RUN /var/lang/bin/python3 -m pip install faiss-cpu
RUN /var/lang/bin/python3 -m pip install opensearch-py

RUN /var/lang/bin/python3 -m pip install -U /var/task/bedrock-sdk/botocore-1.29.162-py3-none-any.whl -t /var/task
RUN rm -rf /var/task/bedrock-sdk

CMD ["lambda_function.lambda_handler"]