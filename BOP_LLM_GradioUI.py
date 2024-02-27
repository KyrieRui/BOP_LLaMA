import gradio as gr
import random
import time
from llama_cpp import Llama
from images import logo_svg

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
use_mlock = True  # Force system to keep model in RAM.
model_path="/Users/zwang/Desktop/pgpt/privateGPT/models/cache/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/blobs/3e0039fd0273fcbebb49228943b17831aadd55cbcbf56f0af00499be2040ccf9"

llm = Llama(
    model_path=model_path,
    chat_format="llama-2",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
)

llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are an assistant who perfectly Answer the question. You don't lie, Not make up facts at all. When faced with a question that you don't know how the answer, you will break it down into multiple parts and answer the part you have answer separately. When faced with a question that is completely unanswerable you will say, I'm sorry I don't know."},
        {
            "role": "user",
            "content": "Describe climate change in New Zealand in 2024."
        }
    ]
)

def respond(message, history):
    result = llm(message, max_tokens=-1)
    bot_message = result['choices'][0]['text']
    return bot_message

demo = gr.ChatInterface(
    fn=respond,
    chatbot=gr.Chatbot(
        label="BOP GPT",
        value=[], 
        height=480
    ),
)

demo.launch()