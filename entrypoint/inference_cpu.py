import os, pdb
import json
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel
from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import torch
import torch_neuron

JSON_CONTENT_TYPE = 'application/json'

def model_fn(model_dir):
    model_name = "google/tapas-base-finetuned-wtq"
    model = TapasForQuestionAnswering.from_pretrained(model_name)
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    return (model, tokenizer)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data

    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return

def predict_fn(input_data, models):
    model_tapas, tokenizer = models
    
    data = input_data[0]["data"]
    queries = input_data[0]["queries"]
    table = pd.DataFrame.from_dict(data)

    inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
    outputs = model_tapas(**inputs)

    
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
    )

    id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    aggregation_predictions_string = [
        id2aggregation[x] for x in predicted_aggregation_indices
    ]

    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            answers.append(table.iat[coordinates[0]])
        else:
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
            answers.append(", ".join(cell_values))

    print("")
    queries_and_answers = []
    for query, answer, predicted_agg in zip(
        queries, answers, aggregation_predictions_string
    ):
        if predicted_agg == "NONE":
            print("Predicted answer: " + answer)
            queries_and_answers.append(f"Query:{query}\nAnswer:{answer}")
        else:
            print("Predicted answer: " + predicted_agg + " > " + answer)
            queries_and_answers.append(f"Query:{query}\nAnswer:{predicted_agg} > {answer}")

    return "\n".join(queries_and_answers)

def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept

    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
