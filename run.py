
import sagemaker
import boto3

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

print(f"sagemaker role arn: {role}")


from sagemaker.huggingface import HuggingFaceModel

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   model_data="s3://hf-sagemaker-inference/model.tar.gz",  # path to your trained sagemaker model
   role=role, # iam role with permissions to create an Endpoint
   transformers_version="4.26", # transformers version used
   pytorch_version="1.13", # pytorch version used
   py_version="py39", # python version of the DLC
)


# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
   initial_instance_count=1,
   instance_type="ml.m5.xlarge"
)

# example request, you always need to define "inputs"
data = {
   "inputs": "The new Hugging Face SageMaker DLC makes it super easy to deploy models in production. I love it!"
}

# request
predictor.predict(data)

# delete endpoint
predictor.delete_model()
predictor.delete_endpoint()
