import gradio as gr
import random
import time
from llama_cpp import Llama
from images import logo_svg

from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
model_path="/Users/zwang/Desktop/pgpt/privateGPT/models/cache/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/blobs/3e0039fd0273fcbebb49228943b17831aadd55cbcbf56f0af00499be2040ccf9"
llm = Llama(
    model_path=model_path,
    chat_format="llama-2",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    use_mlock = True
)

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model = HuggingFaceEmbeddings(
                model_name=embed_model_id,
                model_kwargs={'device': 'mps'}, 
                encode_kwargs={'device': 'mps'}
)

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma
loader = CSVLoader(file_path="./RAG_files/test_demo.csv")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT},
)


def respond(message, history):
    bot_message = qa_chain(message)['result']
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