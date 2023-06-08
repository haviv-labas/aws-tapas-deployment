import os
import json
import tensorflow  # to workaround a protobuf version conflict issue
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

import torch
import torch_neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

JSON_CONTENT_TYPE = 'application/json'
os.environ['NEURON_RT_NUM_CORES']='1'


def model_fn(model_dir):
    tokenizer_init = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
    model_file =os.path.join(model_dir, 'neuron_compiled_model.pt')
    model_neuron = torch.jit.load(model_file)
    return (model_neuron, tokenizer_init)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return example_query

    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return example_query


def predict_fn(input_data, models):
    model_tapas, tokenizer = models
    
    data = input_data["data"]
    queries = input_data["queries"]
    table = pd.DataFrame.from_dict(data)
    inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
    outputs = model_tapas(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
    )

    return "SUCCESSFUL MODEL RUN"

def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept

    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

