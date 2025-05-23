import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype
from pathlib import Path
import os
from torchvision import transforms
from PIL import Image
import numpy as np


class InferenceModule:
    def __init__(self):
        self.url = "localhost:8001"
        self.client = grpcclient.InferenceServerClient(url=self.url)
        self.img_path = os.path.join(Path(__file__).parents[1], "api/imgs/image.jpg")

    def infer_image(
            self,
            model_name="classifier_onnx",
    ):
        
        model_meta, model_config = self.parse_model_metadata(model_name)
        channels, height, width = model_meta.inputs[0].shape[1:]
        dtype = model_meta.inputs[0].datatype

        # preprocess the image
        img = self.preprocess_image()
        
        # create input tensor for Triton
        inputs = [
            grpcclient.InferInput(
                model_meta.inputs[0].name,
                [1, channels, height, width],
                dtype,
            )
        ]
        inputs[0].set_data_from_numpy(img.astype(triton_to_np_dtype(dtype)))

        # define output tensor
        outputs = [
            grpcclient.InferRequestedOutput(
                model_meta.outputs[0].name,
            )
        ]

        # perform inference
        results = self.client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )

        output = results.as_numpy(model_meta.outputs[0].name)[0]
        cls_idx = np.argmax(output)
        cls_logits = output[cls_idx]

        print(f"Predicted class index: {cls_idx}")
        print(f"Predicted class logits: {cls_logits}")


    def parse_model_metadata(self, model_name):
        '''
        Parse the model metadata and configuration.
        '''

        model_metadata = self.client.get_model_metadata(model_name)
        model_config = self.client.get_model_config(model_name)

        return model_metadata, model_config
    
    def preprocess_image(self):

        transforms_custom = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        image = Image.open(self.img_path)
        image = transforms_custom(image)
        image = image.unsqueeze(0)

        return image.numpy()
    