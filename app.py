from fastai.vision.all import *
import gradio as gr

# Cargar el modelo exportado (debe llamarse export.pkl)
learn = load_learner("export.pkl")

def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return {pred: float(probs[pred_idx])}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Clasificador de Osos",
    description="Sube una imagen para clasificarla."
)

# Render necesita host público y puerto fijo
demo.launch(server_name="0.0.0.0", server_port=7860)
