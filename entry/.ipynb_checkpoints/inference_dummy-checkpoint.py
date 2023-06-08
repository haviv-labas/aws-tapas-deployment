import os
import json
import tensorflow  # to workaround a protobuf version conflict issue
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

JSON_CONTENT_TYPE = 'application/json'
os.environ['NEURON_RT_NUM_CORES']='1'


def model_fn(model_dir):
    return (None, None)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data

    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return


def predict_fn(input_data, models):
    return "SUCCESFUL_REQUEST"

def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept

    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

