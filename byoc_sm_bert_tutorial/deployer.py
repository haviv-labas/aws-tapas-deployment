from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
from sagemaker.huggingface.model import HuggingFaceModel

from sagemaker.huggingface import HuggingFace

import pdb

import sagemaker
import boto3


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import TapasTokenizer, TapasForQuestionAnswering




import boto3
import time
from sagemaker.utils import name_from_base
import sagemaker

class Deployer():
    model_to_use: str = "TAPAS"
    algorithm_name="inference-to-deploy"
    def get_model_and_tokeniser(self):
        model_name = "google/tapas-mini-finetuned-wtq"
        self.model = TapasForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        
    def tracing_inputs(self):
        data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
        queries = [
            "What is the name of the first actor?",
            "How many movies has George Clooney played in?",
            "What is the total number of movies?",
        ]
        table = pd.DataFrame.from_dict(data)
        inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
        # outputs = model(**inputs)
        # predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
            #inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
        #)
        example_inputs = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']
        
        return example_inputs

        
    def trace_model(self):
        model_neuron = torch.neuron.trace(model, self.tracing_inputs(), verbose=1, compiler_workdir='./compilation_artifacts', strict=False)
        print(model_neuron.graph)
        model_neuron.save('neuron_compiled_model.pt')
        
    def build_and_upload_to_s3(self):
        os.system("tar -czvf model.tar.gz neuron_compiled_model.pt")
        # upload model to S3
        role = sagemaker.get_execution_role()
        sess=sagemaker.Session()
        region=sess.boto_region_name
        bucket=sess.default_bucket()
        sm_client=boto3.client('sagemaker')
        model_key = '{}/model/model.tar.gz'.format('inf1_compiled_model')
        model_path = 's3://{}/{}'.format(bucket, model_key)
        boto3.resource('s3').Bucket(bucket).upload_file('model.tar.gz', model_key)
        print("Uploaded model to S3:")
        print(model_path)


    def deploy_ecr_image(self):
        import os
        import boto3
        import sagemaker

        role = sagemaker.get_execution_role()
        sess = sagemaker.Session()

        bucket = sess.default_bucket()
        prefix = "inf1_compiled_model/model"

        # Get container name in ECR
        client=boto3.client('sts')
        account=client.get_caller_identity()['Account']

        my_session=boto3.session.Session()
        region=my_session.region_name

        ecr_image='{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, self.algorithm_name)
        print(ecr_image)
        key = os.path.join(prefix, "model.tar.gz")
        pretrained_model_data = "s3://{}/{}".format(bucket, key)
        print(pretrained_model_data)
        from sagemaker.pytorch.model import PyTorchModel

        pytorch_model = PyTorchModel(
            model_data=pretrained_model_data,
            role=role,
            source_dir="code",
            framework_version="1.7.1",
            entry_point="inference.py",
            image_uri=ecr_image
        )

        # Let SageMaker know that we've already compiled the model via neuron-cc
        pytorch_model._is_compiled_model = True
        predictor = pytorch_model.deploy(initial_instance_count=1, instance_type="ml.inf1.2xlarge")
        print(predictor.endpoint_name)
        
        predictor.serializer = sagemaker.serializers.JSONSerializer()
        predictor.deserializer = sagemaker.deserializers.JSONDeserializer()
        
        result = predictor.predict(
            [
                "Never allow the same bug to bite you twice.",
                "The best part of Amazon SageMaker is that it makes machine learning easy.",
            ]
        )
        print(result)
        
        
        predictor.delete_endpoint(predictor.endpoint)