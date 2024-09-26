import time
import torch
import ollama
from time import time
import utils as util

util.start_ollama_server()

curr_Chat = util.Chat_object(title=util.get_chat_title())
# Pre cashing model with a message
models_list = util.get_available_models()
model = util.choose_model(models_list=models_list)
print("Booting Chat...", end="\r")
response = ollama.chat(
    model=model, messages=[{"role": "user", "content": "Don't reply to this message"}]
)
print(curr_Chat.title + " " * 30)
i = 1
while True:
    message_content = util.get_message_content()
    curr_Chat.prompts.append(message_content)
    curr_Chat.messages.append({"role": "user", "content": message_content})
    util.print_message_tab(message_content)
    response, total_time = util.get_response(model=model, curr_Chat=curr_Chat)
    util.print_response_tab(
        model_name=model, response=response["message"]["content"], index=i
    )
    curr_Chat.messages.append(response["message"])
    curr_Chat.replies.append(response["message"]["content"])

    print(f"Computation time {total_time}")
