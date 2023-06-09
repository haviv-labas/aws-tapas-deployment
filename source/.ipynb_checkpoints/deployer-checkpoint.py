import os

class Deployer:
    algorithm_name = "inference-to-deploy"

    def __init__(self, use_neuron=False):
        self.entrypoint_to_use = (
            "inference_neuron.py" if use_neuron else "inference_cpu.py"
        )

    def get_model_and_tokeniser(self):
        raise NotImplemented

    def tracing_inputs(self):
        raise NotImplemented

    def trace_model(self):
        raise NotImplemented

    def upload_model_to_s3(self):
        raise NotImplemented

    def deploy_ecr_image(self):
        raise NotImplemented
    
    def build_ecr_image(self):
        os.system("bash ./build_and_push.sh")
        
    def test_endpoint(self):
        return self.predictor.predict(self.endpoint_testing_query())

    def terminate(self):
        return self.predictor.delete_endpoint(self.predictor.endpoint)

    def endpoint_testing_query(self):
        raise NotImplemented
