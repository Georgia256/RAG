from llama_index.core import StorageContext, ServiceContext, load_index_from_storage
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
#from llama_index.postprocessor.cohere_rerank import CohereRerank
#from llama_index.llms import BaseLanguageModel
import os
import json
from dotenv import load_dotenv
import chainlit as cl

import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          pipeline)
from llama_index.llms.huggingface import HuggingFaceLLM
load_dotenv()

# Define API keys
#GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
config_data = json.load(open("/content/drive/MyDrive/RAG-llamaindex/config.json"))
HF_TOKEN = config_data["HF_TOKEN"]
#COHERE_API_KEY = os.getenv("COHERE_API_KEY")


model_name = "meta-llama/Meta-Llama-3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Initialize the Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name ,token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name ,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)
llm= HuggingFaceLLM(model=model, tokenizer=tokenizer)
@cl.on_chat_start
async def factory():
    storage_context = StorageContext.from_defaults(persist_dir="./storage")

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm,
                        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )
    #cohere_rerank = CohereRerank(api_key=COHERE_API_KEY, top_n=2)

    index = load_index_from_storage(storage_context, service_context=service_context)

    query_engine = index.as_query_engine(
        service_context=service_context,
        similarity_top_k=10,
        #node_postprocessors=[cohere_rerank],
        # streaming=True,
    )

    cl.user_session.set("query_engine", query_engine)

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  
    response = await cl.make_async(query_engine.query)(message.content)

    response_message = cl.Message(content="")

    for token in response.response:
        await response_message.stream_token(token=token)

    await response_message.send()
