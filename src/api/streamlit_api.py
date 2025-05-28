import streamlit as st
import os


class StreamlitForm:
    def __init__(
            self,
            classes_data, 
            title="Triton test",
            img_path="src/api/imgs"):
        self.classes_data = classes_data
        self.title = title
        self.img_path = img_path

    def get_stremlit_form(self, triton_infer):
        
        st.title(self.title)

        uploaded_file = st.file_uploader("Upload image")

        if uploaded_file is not None:
            file_path = os.path.join(self.img_path, "image.jpg")

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"File saved!")

        if st.button("Check info"):
            cls_idx, cls_logits = triton_infer.infer_image()
            st.text(f"class: {self.classes_data[cls_idx]}\nconfidence: {cls_logits}")
