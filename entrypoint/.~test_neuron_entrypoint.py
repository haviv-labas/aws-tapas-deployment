from inference_neuron import model_fn, predict_fn
import os

os.environ["NEURON_RT_NUM_CORES"] = "1"


def test_cpu_entrypoint():
    model_neuron = model_fn("./")
    input_data = [
        {
            "data": {
                "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
                "Number of movies": ["87", "53", "69"],
            },
            "queries": [
                "What is the name of the first actor?",
                "How many movies has George Clooney played in?",
                "What is the total number of movies?",
            ],
        }
    ]

    assert istype(list, predict_fn(input_data, model_neuron))
