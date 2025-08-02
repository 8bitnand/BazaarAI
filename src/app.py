import os
import gradio as gr
from agent import BazarAgent

MODEL_LIST = ["mistral:latest", "deepseek-r1:14b", "Qwen/Qwen3-1.7B"]
bseagent = BazarAgent()
fallback_model = bseagent.get_model("mistral:latest")
agent = bseagent.initilize_any_agent(fallback_model)


def respond(user_msg, history, model_selector):
    global agent, fallback_model
    if model_selector != bseagent.curr_model:
        fallback_model = bseagent.get_model(model_selector)
        agent = bseagent.initilize_any_agent(fallback_model)

    history = history or []
    try:
        output = agent.invoke(user_msg)
        response = fallback_model.invoke(
            f"You are a json converter.\
                your Role is to convert unstructured data or json into simple human understandable structured markdown format. \
                Only output the final markdown output.\
                Do not return explanations, reasoning thinking proceess. if you print anything else there will be a penalty.\
                if the the below is a python dict or Json  convert this to a valid markdown format. Else return as it is\
                {output['output']}"
        )
        history.append([user_msg, response.content])
    except Exception as e:
        history.append([user_msg, e])

    yield history, ""


def build_ui():
    demo = gr.Blocks(theme=gr.themes.Origin()).queue()
    with demo:
        gr.Markdown("# BazarAI  - know what is happning in Indian stock market (BSE)")
        chatbot = gr.Chatbot(elem_id="chatbot")

        # with gr.Accordion("Thinking", open=True):
        #     thinking_output = gr.Markdown("", elem_id="thinking-box")

        with gr.Row():
            with gr.Column(scale=4):
                user_input = gr.Textbox(
                    placeholder="Ask questions related to BSE",
                    show_label=False,
                    elem_id="user-input",
                )
                # gr.ClearButton(components=user_input)

            model_selector = gr.Dropdown(
                MODEL_LIST,
                value=MODEL_LIST[0],
                show_label=False,
                elem_id="model-select",
            )

        # Hook up submit event
        user_input.submit(
            fn=respond,
            inputs=[user_input, chatbot, model_selector],
            outputs=[chatbot, user_input],
        )
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=False)


# {
#     "input": "hi",
#     "chat_history": [
#         HumanMessage(content="hi", additional_kwargs={}, response_metadata={}),
#         AIMessage(
#             content=' Action:\n  ```\n  {\n    "action": "Final Answer",\n    "action_input": "Hello! How can I help you with stocks today?"\n  }\n\n  ',
#             additional_kwargs={},
#             response_metadata={},
#         ),
#     ],
#     "output": ' Action:\n  ```\n  {\n    "action": "Final Answer",\n    "action_input": "Hello! How can I help you with stocks today?"\n  }\n\n  ',
#     "intermediate_steps": [],
# }
