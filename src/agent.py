# from langchain.llms import LlamaCpp
from langchain.agents import initialize_agent, AgentType
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from tools import company_scrip_code, company_details, top_gainers, top_losers
from langchain_ollama import ChatOllama
import torch
from langchain.memory import ConversationBufferMemory


class BazarAgent:
    def __init__(self) -> None:
        self.ollama_models = ["mistral:latest", "deepseek-r1:14b"]
        self.hf_models = ["Qwen/Qwen3-1.7B"]
        self.curr_model = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="output", return_messages=True
        )
        self.tools = [company_details, company_scrip_code, top_gainers, top_losers]

    def load_hf_model(self, model_name: str, **kwargs):

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="src/models")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="src/models",
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **kwargs,
            # temperature=0.7,
            # top_k=20,
            # top_p=0.8,
            # min_p=0,
        )
        llm = HuggingFacePipeline(
            pipeline=pipe, pipeline_kwargs={"enable_thinking": True}
        )

        return llm

    def load_ollama_model(self, model_name: str, **kwargs):
        llm = ChatOllama(model=model_name, **kwargs)
        return llm

    def initilize_any_agent(self, llm):

        agent = initialize_agent(
            self.tools,
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            memory=self.memory,
        )
        return agent

    def get_model(self, model_selector):
        self.curr_model = model_selector

        if model_selector in self.hf_models:
            return self.load_hf_model(model_selector)
        elif model_selector in self.ollama_models:
            return self.load_ollama_model(model_selector)
        # handle exceptions or use fallback model


# "500256": "LOK HOUSING & CONSTRUCTIONS LTD."


# bagent = BazarAgent()
# agent = bagent.initilize_any_agent(bagent.load_ollama_model("mistral:latest"))
# # # # agent = bagent.initilize_any_agent(bagent.load_hf_model("Qwen/Qwen3-1.7B"))
# # print(agent.invoke("Get details of the company Procter and Gamble"))
# for chunk in agent.stream("Get details of the company Procter and Gamble"):
#     print("strm", chunk)
