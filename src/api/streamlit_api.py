import streamlit as st
import os


class StreamlitForm:
    def __init__(
            self, 
            title="Triton test",
            img_path="src/api/imgs"):
        self.title = title
        self.img_path = img_path

    def get_stremlit_form(self, triton_infer):
        st.title(self.title)

        if st.button("check model config"):
            triton_infer.infer_image()

        uploaded_file = st.file_uploader("Upload image")

        if uploaded_file is not None:
            file_path = os.path.join(self.img_path, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"File saved: {file_path}")

        st.text("test")
