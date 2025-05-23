from src.api.streamlit_api import StreamlitForm
from src.triton_inference.infer_triton import InferenceModule


streamlit_obj = StreamlitForm()
triton_infer = InferenceModule()


def main():

    streamlit_obj.get_stremlit_form(
        triton_infer=triton_infer,
    )

if __name__ == "__main__":
    main()
