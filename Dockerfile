FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY lambda_function.py ./
COPY fan_kmeans_model.pkl ./  
COPY motor_kmeans_model.pkl ./  

CMD ["lambda_function.lambda_handler"]