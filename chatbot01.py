from langchain.document_loaders import WebBaseLoader
from fastapi.responses import JSONResponse
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr


n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
use_mlock = True  # Force system to keep model in RAM.

# Make sure the model path is correct for your system!
model_path="/Users/zwang/Desktop/pgpt/privateGPT/models/cache/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/blobs/3e0039fd0273fcbebb49228943b17831aadd55cbcbf56f0af00499be2040ccf9"

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

def predict(prompt, history):
    return llm(prompt)

demo = gr.ChatInterface(
    fn=predict,
    title="BOP GPT",
    chatbot = gr.Chatbot(height=300),
)

demo.launch()