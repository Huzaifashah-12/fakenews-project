FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY app.py ${LAMBDA_TASK_ROOT}
COPY model.pkl ${LAMBDA_TASK_ROOT}
COPY vec.pkl ${LAMBDA_TASK_ROOT}

# Set the Lambda handler function
CMD [ "app.lambda_handler" ]
