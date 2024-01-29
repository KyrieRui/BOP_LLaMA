from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from langchain.llms import LlamaCpp
from langchain.document_loaders import WebBaseLoader
from fastapi.responses import JSONResponse

from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path="./RAG_files/Test_demo.csv")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
use_mlock = True  # Force system to keep model in RAM.

# Make sure the model path is correct for your system!
model_path = "./models/llama-2-7b-chat.Q4_K_M.gguf"

embedding = LlamaCppEmbeddings(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    use_mlock=True,
)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
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

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate


app = FastAPI()

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 创建一个列表，用于保存推断结果
inference_results = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceRequest(BaseModel):
    prompt: str


# 创建响应模型
class InferenceResponse(BaseModel):
    data: str  # 这里根据实际情况可能需要调整


prompt_template = """You are helpful. Use the following pieces of context to answer the question at the end. You can answer questions about the provided context.

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


# 定义推断路由
@app.post("/complete")
async def perform_inference(request: InferenceRequest):
    # 在处理请求时记录信息
    logger.info(f"Received prompt: {request.prompt}")

    # 进行推断
    output = qa_chain({"query": request.prompt})

    # 将推断结果保存到列表中
    inference_results.append(output)

    # 返回实际的推断结果给客户端
    return {"data": output}


# 定义获取完整推断信息的路由
@app.get("/complete_results")
async def get_complete_results():
    # 返回已保存的所有推断结果
    return {"data": inference_results}


# 如果需要清空已保存的推断结果，可以定义一个清空路由
@app.delete("/clear_results")
async def clear_results():
    # 清空推断结果列表
    inference_results.clear()
    return {"message": "Inference results cleared"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
