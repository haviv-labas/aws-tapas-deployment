from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch
import torch_neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel
import pandas as pd

model_name = "google/tapas-mini-finetuned-wtq"
#model_name = "google/tapas-mini"
model = TapasForQuestionAnswering.from_pretrained(model_name)
tokenizer = TapasTokenizer.from_pretrained(model_name, return_dict=False)

data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
queries = [
    "What is the name of the first actor?",
    "How many movies has George Clooney played in?",
    "What is the total number of movies?",
]


# tapas_mini_query = "[CLS] Sentence [SEP] Flattened table [SEP]"


table = pd.DataFrame.from_dict(data)
inputs = tokenizer(table=table, queries=queries, max_length=128, padding="max_length", return_tensors="pt")
#max_length=128
#inputs = tokenizer(tapas_mini_query, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

outputs = model(**inputs)
predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
)

example_inputs = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']
model_neuron = torch.neuron.trace(model, example_inputs, verbose=1, compiler_workdir='./compilation_artifacts', strict=False)


model_neuron.save('neuron_compiled_model.pt')


os.system("tar -czvf model.tar.gz neuron_compiled_model.pt")
