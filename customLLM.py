import torch
from transformers import pipeline
from typing import Optional, List, Mapping, Any

from llama_index import (
    ServiceContext, 
    SimpleDirectoryReader, 
    LangchainEmbedding, 
    ListIndex
)
from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CustomLLM, 
    CompletionResponse, 
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_completion_callback


# set context window size
context_window = 2048
# set number of output tokens
num_output = 256

# store the pipeline/model outisde of the LLM class to avoid memory issues
model_name = "facebook/opt-iml-max-30b"
pipeline = pipeline("text-generation", model=model_name, device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})

class OurLLM(CustomLLM):

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=context_window,
            num_output=num_output,
            model_name=model_name
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt_length = len(prompt)
        response = pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]

        # only return newly generated tokens
        text = response[prompt_length:]
        return CompletionResponse(text=text)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()

# define our LLM
llm = OurLLM()

service_context = ServiceContext.from_defaults(
    llm=llm, 
    context_window=context_window, 
    num_output=num_output
)

# Load the your data
documents = SimpleDirectoryReader('./data').load_data()
index = ListIndex.from_documents(documents, service_context=service_context)

# Query and print response
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")
print(response)