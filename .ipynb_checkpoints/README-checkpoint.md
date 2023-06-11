![SageMaker](https://github.com/aws/sagemaker-inference-toolkit/raw/master/branding/icon/sagemaker-banner.png)

# SageMaker Inference Toolkit

[![Latest Version](https://img.shields.io/pypi/v/sagemaker-inference.svg)](https://pypi.python.org/pypi/sagemaker-inference) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker-inference.svg)](https://pypi.python.org/pypi/sagemaker-inference) [![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)

# TAPAS deployment via Sagemaker-Neuron.
Please see ```run.ipynb``` for full details on how to run and test this repository.


## Overview
This notebook creates an instance of ```TAPAS_Deployer``` and calls all neccessary actions to build, deploy, and test a mini variant of TAPAS for tabular question answering. For details, please refer to the source files included in ```./source``` and ```./entrypoint``` which were refactored to be easy to read.

## How to use this notebook..
- Create an AWS account.
- Create an IAM role with the following access permissions: ```AmazonSageMakerFullAccess, EC2InstanceProfileForImageBuilderECRContainerBuilds, AWSAppRunnerServicePolicyForECRAccess```
- Start a new Notebook instance in Sagemaker using the role created above.
- Clone this repository and run the notebook ```run.ipynb```.


