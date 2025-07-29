# from langchain.llms import LlamaCpp
from langchain.agents import initialize_agent, AgentType
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from tools import company_details

import torch
from langchain.memory import ConversationBufferMemory


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", cache_dir="src/models")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    cache_dir="src/models",
    torch_dtype=torch.float32,
    device_map="cpu",
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.7)
llm = HuggingFacePipeline(pipeline=pipe)

# tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
# mdl = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", torch_dtype="auto")
# pipe = pipeline("text-generation", model=mdl, tokenizer=tok, temperature=0.7)
# llm  = HuggingFacePipeline(pipeline=pipe)


tools = [company_details]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
)


def run_agent(user_input: str) -> str:
    system_prompt = f""" Your are a agent that can call functions and talk naturaly.When you decide to call the tool, output _only_:
    Thought: <your reasoning>
    Action: get_company_scrip_code
    Action Input: <company name>

    When you know the answer, output _only_:
    Thought: I now know the final answer
    Final Answer: <scrip code>

    User Question: {user_input}
    """
    return agent.run(user_input)


run_agent("what is the scrip code of company LOK HOUSING & CONSTRUCTIONS LTD")
