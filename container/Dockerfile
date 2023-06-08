# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-neuron:1.7.1-neuron-py36-ubuntu18.04
# FROM 763104351884.dkr.ecr.eu-north-1.amazonaws.com/djl-inference:0.22.1-neuronx-sdk2.10.0
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.22.1-neuronx-sdk2.10.0
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuronx:1.13.0-transformers4.28.1-neuronx-py38-sdk2.9.1-ubuntu20.04
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuron:1.10.2-transformers4.20.1-neuron-py37-sdk1.19.1-ubuntu18.04

RUN pip install "transformers==4.7.0"
RUN pip install "pandas==1.3.5"

RUN pip install --upgrade --no-cache-dir torch-neuron neuronx-cc[tensorflow] torchvision torch --extra-index-url=https://pip.repos.neuron.amazonaws.com
# RUN pip install --upgrade --no-cache-dir torch-neuron neuron-cc[tensorflow] torchvision torch --extra-index-url=https://pip.repos.neuron.amazonaws.com
RUN pip install --upgrade --no-cache-dir 'transformers==4.6.0'
RUN pip install "torch-scatter==2.1.1"

# RUN apt-get remove aws-neuron-dkms
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -
RUN apt-get update
#RUN apt-get install fakeroot build-essential crash kexec-tools makedumpfile kernel-wedge
#RUN apt-get build-dep linux
#RUN apt-get install git-core libncurses5 libncurses5-dev libelf-dev asciidoc binutils-dev
RUN apt update
# RUN apt install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
RUN apt-get install -y aws-neuronx-dkms

# torch-scatter==2.1.1
# pandas==1.5.2
# CMD ["/usr/local/bin/entrypoint.sh"]

