```
./server -m ./models/llama-2-7b-chat.Q4_K_M.gguf

./main -m ./models/llama-2-7b-chat.Q4_K_M.gguf -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt

./main -m models/llama-2-7b-chat.Q4_K_M.gguf -p "what is GPT" -n 400 -e

```

```
conda activate llm

python mian.py

python main_excel.py

python main_bus.py
```

conda create -n llm python=3.11
conda activate llm
