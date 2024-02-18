import streamlit as st
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

model_path="/Users/zwang/Desktop/pgpt/privateGPT/models/cache/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/blobs/3e0039fd0273fcbebb49228943b17831aadd55cbcbf56f0af00499be2040ccf9"
model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

st.set_page_config(page_title="DOCAI", page_icon="🤖", layout="wide", )
st.markdown(f"""

""", unsafe_allow_html=True)


def write_text_file(content, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(content) 
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
    return False

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
use_mlock = True  # Force system to keep model in RAM.
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
    use_mlock = True
)

from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

embed_model = HuggingFaceEmbeddings(
                model_name=embed_model_id,
                model_kwargs={'device': 'mps'}, 
                encode_kwargs={'device': 'mps'}
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

st.title("📄 Document Conversation 🤖")
uploaded_file = st.file_uploader("Upload an article", type="txt")

if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')
    # st.write(content)
    file_path = "./RAG_files"
    write_text_file(content, file_path)

loader = TextLoader(file_path)
docs = loader.load()    
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
db = Chroma.from_documents(texts, embed_model)    
st.success("File Loaded Successfully!!")

# Query through LLM    
question = st.text_input("Ask something from the file", placeholder="in the text?", disabled=not uploaded_file,)    
if question:
    similar_doc = db.similarity_search(question, k=1)
    context = similar_doc[0].page_content
    query_llm = LLMChain(llm=llm, prompt=prompt)
    response = query_llm.run({"context": context, "question": question})        
    st.write(response)        