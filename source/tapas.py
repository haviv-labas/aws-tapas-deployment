from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
from sagemaker.huggingface.model import HuggingFaceModel

from sagemaker.huggingface import HuggingFace

import pdb

import sagemaker
import boto3


class Deployer:
    model_to_use: str = "TAPAS"

    def run_everything(self):
        model_name = "google/tapas-base-finetuned-wtq"
        model = TapasForQuestionAnswering.from_pretrained(model_name)
        tokenizer = TapasTokenizer.from_pretrained(model_name)

        data = {
            "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
            "Number of movies": ["87", "53", "69"],
        }
        queries = [
            "What is the name of the first actor?",
            "How many movies has George Clooney played in?",
            "What is the total number of movies?",
        ]
        table = pd.DataFrame.from_dict(data)
        inputs = tokenizer(
            table=table, queries=queries, padding="max_length", return_tensors="pt"
        )
        outputs = model(**inputs)
        (
            predicted_answer_coordinates,
            predicted_aggregation_indices,
        ) = tokenizer.convert_logits_to_predictions(
            inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
        )

        # let's print out the results:
        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
        aggregation_predictions_string = [
            id2aggregation[x] for x in predicted_aggregation_indices
        ]

        answers = []
        for coordinates in predicted_answer_coordinates:
            if len(coordinates) == 1:
                # only a single cell:
                answers.append(table.iat[coordinates[0]])
            else:
                # multiple cells
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(table.iat[coordinate])
                answers.append(", ".join(cell_values))

        # display(table)

        print("")
        for query, answer, predicted_agg in zip(
            queries, answers, aggregation_predictions_string
        ):
            print(query)
            if predicted_agg == "NONE":
                print("Predicted answer: " + answer)
            else:
                print("Predicted answer: " + predicted_agg + " > " + answer)

        return table

    def deploy(self):
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            iam = boto3.client("iam")
            role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

        # Hub Model configuration. <https://huggingface.co/models>
        hub = {
            "HF_MODEL_ID": "distilbert-base-uncased-distilled-squad",  # model_id from hf.co/models
            "HF_TASK": "question-answering",  # NLP task you want to use for predictions
        }

        # create Hugging Face Model Class
        huggingface_model = HuggingFaceModel(
            env=hub,  # configuration for loading model from Hub
            role=role,  # iam role with permissions to create an Endpoint
            transformers_version="4.6",  # transformers version used
            py_version="py36",
            pytorch_version="1.7",  # pytorch version used
        )

        # deploy model to SageMaker Inference
        predictor = huggingface_model.deploy(
            initial_instance_count=1, instance_type="ml.m5.xlarge"
        )

        # example request, you always need to define "inputs"
        data = {
            "inputs": {
                "question": "What is used for inference?",
                "context": "My Name is Philipp and I live in Nuremberg. This model is used with sagemaker for inference.",
            }
        }

        # request
        predictor.predict(data)

        print(predictor.predict(data))

    def deploy_tapas(self):
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            iam = boto3.client("iam")
            role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]
        # Hub Model configuration. https://huggingface.co/models
        hub = {
            "HF_MODEL_ID": "google/tapas-base-finetuned-wtq",
            "HF_TASK": "table-question-answering",
        }

        hub = {
            "HF_MODEL_ID": "google/tapas-small-finetuned-wtq",
            "HF_TASK": "table-question-answering",
        }

        # 058095970122.dkr.ecr.eu-north-1.amazonaws.com/pytorch-extending-our-containers-cifar10-example
        algorithm_name = "pytorch-extending-our-containers-cifar10-example"
        # ecr_image = "{}.dkr.ecr.{}.amazonaws.com/{}:latest".format(account, region, algorithm_name)
        # ecr_image = "058095970122.dkr.ecr.eu-north-1.amazonaws.com/pytorch-extending-our-containers-cifar10-example"
        # ecr_image = "058095970122.dkr.ecr.eu-north-1.amazonaws.com/pytorch-extending-our-containers-cifar10-example:latest"
        # ecr_image = "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04"
        # ecr_image = "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
        ecr_image = "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-inference-neuronx:1.13.0-transformers4.28.1-neuronx-py38-sdk2.9.1-ubuntu20.04"
        print(f"Using IMAGE: {ecr_image}")

        # configure git settings
        # git_config = {'repo': 'https://github.com/huggingface/transformers.git','branch': 'v4.4.2'} # v4.4.2 refers to the transformers_version you use in the estimator
        git_config = {
            "repo": "https://github.com/haviv-labas/scrub.git",
            "branch": "main",
        }  # v4.4.2 refers to the transformers_version you use in the estimator

        #!@#@!#@!#@!#
        # create Hugging Face Model Class
        # entry_point='/foo.py',

        huggingface_model = HuggingFaceModel(
            entry_point="foo.py",
            source_dir="./foo",
            git_config=git_config,
            transformers_version="4.6",
            pytorch_version="1.7",
            py_version="py36",
            image_uri=ecr_image,
            env=hub,
            role=role,
        )

        # huggingface_model.

        # deploy model to SageMaker Inference
        predictor = huggingface_model.deploy(
            entry_point="/home/ec2-user/SageMaker/scrub/foo.py",
            source_dir="/home/ec2-user/SageMaker/scrub/foo",
            initial_instance_count=1,  # number of instances
            instance_type="ml.m5.xlarge",  # ec2 instance type
        )

        output = predictor.predict(
            {
                "inputs": {
                    "query": "How many stars does the transformers repository have?",
                    "table": {
                        "Repository": ["Transformers", "Datasets", "Tokenizers"],
                        "Stars": ["36542", "4512", "3934"],
                        "Contributors": ["651", "77", "34"],
                        "Programming language": [
                            "Python",
                            "Python",
                            "Rust, Python and NodeJS",
                        ],
                    },
                },
            }
        )

        print(output)
        return output

    def train_tapas(self):
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            iam = boto3.client("iam")
            role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

        hyperparameters = {
            "model_name_or_path": "google/tapas-large-finetuned-wtq",
            "output_dir": "s3://sagemaker-eu-north-1-0580959/model.tar.gz"
            # add your remaining hyperparameters
            # more info here https://github.com/huggingface/transformers/tree/v4.26.0/path/to/script
        }

        # git configuration to download our fine-tuning script
        git_config = {
            "repo": "https://github.com/huggingface/transformers.git",
            "branch": "v4.26.0",
        }

        # creates Hugging Face estimator
        huggingface_estimator = HuggingFace(
            entry_point="/home/ec2-user/SageMaker/scrub/foo.py",
            source_dir="/home/ec2-user/SageMaker/scrub/foo",
            instance_type="ml.m5.xlarge",
            instance_count=1,
            role=role,
            git_config=git_config,
            transformers_version="4.6",
            pytorch_version="1.7",
            py_version="py36",
            hyperparameters=hyperparameters,
        )

        # starting the train job
        huggingface_estimator.fit()
