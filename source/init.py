import gradio
from main import response

gradio.ChatInterface(response).launch()
