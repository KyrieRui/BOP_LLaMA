import gradio as gr
import random
import time

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp


with gr.Blocks() as demo:
    n_gpu_layers = 1  # Metal set to 1 is enough.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    use_mlock = True  # Force system to keep model in RAM.  

    model_path="/Users/zwang/Desktop/pgpt/privateGPT/models/cache/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/blobs/3e0039fd0273fcbebb49228943b17831aadd55cbcbf56f0af00499be2040ccf9"


    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=2048,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        verbose=True,
        use_mlock = True
    )

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = llm(message)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()

