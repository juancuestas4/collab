from fastai.vision.all import *
import gradio as gr
import requests
import os

# URL directa del modelo en Google Drive (DEBE ser un link con "uc?export=download")
url = "https://drive.usercontent.google.com/download?id=1ye9fV7qCz60E3aFMBCdom1IiCqDo-sHl&export=download&authuser=0"

# Descargar el modelo solo si no existe
if not os.path.exists("export.pkl"):
    print("Descargando modelo desde Google Drive...")
    response = requests.get(url)
    with open("export.pkl", "wb") as f:
        f.write(response.content)
    print("Modelo descargado.")

# Cargar modelo
learn = load_learner("export.pkl")

# Función de predicción
def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return {pred: float(probs[pred_idx])}

# Interfaz Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Clasificador de Osos",
    description="Sube una imagen para clasificarla."
)

# Requerido por Render
demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)
