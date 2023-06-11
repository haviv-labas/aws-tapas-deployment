FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-neuron:1.7.1-neuron-py36-ubuntu18.04

RUN pip install "pandas==1.1.5"
RUN pip install --upgrade --no-cache-dir torch-neuron neuron-cc[tensorflow] torchvision torch torch-scatter --extra-index-url=https://pip.repos.neuron.amazonaws.com
RUN pip install --upgrade --no-cache-dir 'transformers==4.6.0'