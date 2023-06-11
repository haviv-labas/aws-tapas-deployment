import os
from abc import ABC, abstractmethod
from typing import Any


class Deployer(ABC):
    algorithm_name: str = "inference-to-deploy"

    def __init__(self, use_neuron: bool = False) -> None:
        self.entrypoint_to_use: str = (
            "inference_neuron.py" if use_neuron else "inference_cpu.py"
        )

    @abstractmethod
    def get_model_and_tokeniser(self) -> None:
        pass

    @abstractmethod
    def tracing_inputs(self) -> None:
        pass

    @abstractmethod
    def trace_model(self) -> None:
        pass

    @abstractmethod
    def upload_model_to_s3(self) -> None:
        pass

    @abstractmethod
    def deploy_ecr_image(self) -> None:
        pass

    def build_ecr_image(self) -> None:
        os.system("bash ./build_and_push.sh")

    def test_endpoint(self) -> Any:
        return self.predictor.predict(self.endpoint_testing_query())

    def terminate(self) -> None:
        return self.predictor.delete_endpoint(self.predictor.endpoint)

    @abstractmethod
    def endpoint_testing_query(self) -> None:
        pass
