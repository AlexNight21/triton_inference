from src.api.streamlit_api import StreamlitForm
from src.triton_inference.infer_triton import InferenceModule


triton_infer = InferenceModule()


def get_classes_data(file_path="./data/imagenet_classes.txt"):
    with open(file_path, "r") as imgnet_cls:
        imgnet_cls_data = imgnet_cls.readlines()
        
    cls_dict = {}

    for data in imgnet_cls_data:
        dat_key, dat_val = data.split(":")
        dat_val = dat_val.strip().replace("'", "")
        if dat_val[-1] == ",":
            dat_val = dat_val[:-1]
        
        cls_dict[int(dat_key.strip())] = dat_val.strip()

    return cls_dict


def main():

    cls_dict = get_classes_data()
    streamlit_obj = StreamlitForm(classes_data=cls_dict)

    streamlit_obj.get_stremlit_form(
        triton_infer=triton_infer,
    )

if __name__ == "__main__":
    main()
