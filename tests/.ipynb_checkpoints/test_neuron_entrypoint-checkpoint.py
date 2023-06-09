from entrypoint.inference_neuron import model_fn, predict_fn


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
