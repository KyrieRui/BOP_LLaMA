from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import os
import io
import gradio as gr
import time
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import fitz
from PIL import Image

custom_prompt_template = """
You are an AI Coding Assitant and your task is to solve coding problems and return code snippets based on given user's query. Below is the user's query.
Query: {query}

You just return the helpful code.
Helpful Answer:
"""

def __init__(self):
        """
        Initialize the PDFChatBot instance.

        Parameters:
            config_path (str): Path to the configuration file (default is "../config.yaml").
        """
        self.processed = False
        self.page = 0
        self.chat_history = []
       
        # Initialize other attributes to None
        self.prompt = None
        self.documents = None
        self.embeddings = None
        self.vectordb = None
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.chain = None

def set_custom_prompt(self):
    prompt = PromptTemplate(template=custom_prompt_template,
    input_variables=['query'])
    return self.prompt

def add_text(self, history, text):
        """
        Add user-entered text to the chat history.

        Parameters:
            history (list): List of chat history tuples.
            text (str): User-entered text.

        Returns:
            list: Updated chat history.
        """
        if not text:
            raise gr.Error('Enter text')
        history.append((text, ''))
        return history

def load_embeddings(self):
        """
        Load embeddings from Hugging Face and set in the config file.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="'sentence-transformers/all-MiniLM-L6-v2'",
            model_kwargs={'device': 'mps'}, 
            encode_kwargs={'device': 'mps'}
        )


#Loading the model
def load_model(self):
    # Load the locally downloaded model here
    model_path="/Users/zwang/Desktop/pgpt/privateGPT/models/cache/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/blobs/3e0039fd0273fcbebb49228943b17831aadd55cbcbf56f0af00499be2040ccf9"

    llm = LlamaCpp(
        model_path = model_path,
        n_gpu_layers = 1,
        n_batch = 512,
        n_ctx=2048,
        f16_kv=True,
        temperature = 0.2,
    )

    return self.model == llm


def load_vectordb(self):
    """
    Load the vector database from the documents and embeddings.
    """
    self.vectordb = Chroma.from_documents(self.documents, self.embeddings)
print("Vector database loaded successfully!")

def chain_pipeline(self):
    llm = load_model()
    qa_prompt = set_custom_prompt()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=self.vectordb.as_retriever(),
        chain_type_kwargs={"prompt": qa_prompt},
    )
    
    return self.chain == qa_chain



def process_file(self, file):
    """
    Process the uploaded PDF file and initialize necessary components: Tokenizer, VectorDB and LLM.

    Parameters:
        file (FileStorage): The uploaded PDF file.
    """
    self.create_prompt_template()
    self.documents = PyPDFLoader(file.name).load()
    self.load_embeddings()
    self.load_vectordb()
    self.load_model()
    self.create_chain()

def generate_response(self, history, query, file):
    """
    Generate a response based on the user's query.

    Parameters:
        history (list): List of chat history tuples.
        query (str): The user's query.
        file (FileStorage): The uploaded PDF file.

    Returns:
        str: The AI-generated response.
    """
    if not query:
        raise gr.Error('Enter a query')
    if not file:
        raise gr.Error('Upload a file')
    self.chat_history = add_text(history, query)
    self.process_file(file)
    response = self.chain_pipeline().generate_response(self.chat_history)
    return response

def render_file(self, file):
    """
    Renders a specific page of a PDF file as an image.

    Parameters:
        file (FileStorage): The PDF file.

    Returns:
        PIL.Image.Image: The rendered page as an image.
    """
    doc = fitz.open(file.name)
    page = doc[self.page]
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image

def generate_response(self, history, query, file):
    """
    Generate a response based on the user's query.

    Parameters:
        history (list): List of chat history tuples.
        query (str): The user's query.
        file (FileStorage): The uploaded PDF file.

    Returns:
        str: The AI-generated response.
    """
    if not query:
        raise gr.Error('Enter a query')
    if not file:
        raise gr.Error('Upload a file')
    self.chat_history = add_text(history, query)
    self.process_file(file)
    response = self.chain_pipeline().generate_response(self.chat_history)
    return response