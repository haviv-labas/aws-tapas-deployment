FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuron:1.10.2-transformers4.20.1-neuron-py37-sdk1.19.1-ubuntu18.04

RUN pip install "transformers==4.7.0"
RUN pip install "pandas==1.3.5"
RUN pip install --upgrade --no-cache-dir torch-neuron neuronx-cc[tensorflow] torchvision torch --extra-index-url=https://pip.repos.neuron.amazonaws.com
RUN pip install --upgrade --no-cache-dir 'transformers==4.6.0'
RUN pip install "torch-scatter==2.1.1"
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -
RUN apt-get update
RUN apt-get install -y aws-neuron-dkms