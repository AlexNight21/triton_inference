import tritonclient.grpc as grpcclient


class InferenceModule:
    def __init__(self):
        self.url = "localhost:8001"
        self.client = grpcclient.InferenceServerClient(url=self.url)

    def infer_image(
            self,
            model_name="classifier_onnx",
    ):
        
        model_meta, model_config = self.parse_model_metadata(model_name)

        print(f"model meta:\n{model_meta.inputs[0].shape}")
        # print(f"model config:\n{model_config}")

    def parse_model_metadata(self, model_name):
        '''
        Parse the model metadata and configuration.
        '''

        model_metadata = self.client.get_model_metadata(model_name)
        model_config = self.client.get_model_config(model_name)

        return model_metadata, model_config
    