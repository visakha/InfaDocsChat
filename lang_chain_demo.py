from langchain_community.llms import HuggingFacePipeline
import os
import torch


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

print(f'CUDA device count: {torch.cuda.device_count()}')
print(f'CUDA device name: {torch.cuda.get_device_name("cuda:0")}')
# print(f'CUDA device name: {torch.cuda.get_device_name("cuda:3")}')
avail_gpu_device=0

hf = HuggingFacePipeline.from_model_id(
    # model_id="microsoft/DialoGPT-medium", task="text-generation", pipeline_kwargs={"max_new_tokens": 200, "pad_token_id": 50256}
    model_id="distilbert-base-cased-distilled-squad", task="text-generation", pipeline_kwargs={"max_new_tokens": 200, "pad_token_id": 50256}
    , device=avail_gpu_device
)

from langchain.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is the best way to buy happiness ?"

print(chain.invoke({"question": question}))

print("completed ... you can terminate this app")