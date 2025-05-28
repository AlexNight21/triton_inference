# triton_inference

<pre>docker compose -f docker-compose.yml -p triton_compose up -d triton prometheus grafana</pre>

<pre>streamlit run main.py</pre>

<pre>trtexec --onnx=model.onnx --saveEngine=model.plan --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:16x3x224x224 --fp16 --useSpinWait --outputIOFormats=fp16:chw --inputIOFormats=fp16:chw</pre>
