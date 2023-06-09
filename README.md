# TAPAS deployment via Sagemaker-Neuron.
Please see ```run.ipynb``` for full details on how to run and test this repository.


## Overview
This notebook creates an instance of ```TAPAS_Deployer``` and calls all neccessary actions to build, deploy, and test a mini variant of TAPAS for tabular question answering. For details, please refer to the source files included in ```./source``` and ```./entrypoint``` which were refactored to be easy to read.

## How to use this notebook..
- Create an AWS account.
- Create an IAM role with the following access permissions: ```AmazonSageMakerFullAccess, EC2InstanceProfileForImageBuilderECRContainerBuilds, AWSAppRunnerServicePolicyForECRAccess```
- Start a new Notebook instance in Sagemaker using the role created above.
- Clone this repository and run this notebook.


