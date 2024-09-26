# Imports
import ollama
import json
import time
import subprocess


# Backend classes and methods for the desktop app
class Chat_object:
    def __init__(self,title:str):
        self.prompts = []
        self.replies = []
        self.reply_time = []
        self.title = title
        self.messages = []



# --------------------------------------
def get_available_models():
    models_dict = ollama.list()
    with open('models_json.json','w') as models_file:
        models_file.write(json.dumps(models_dict))
    # Parsing model names
    model_list = []
    for model in models_dict['models']:
        model_list.append(model['name'])

    return model_list

# --------------------------------------

def print_message_tab(message_content):
    print('Prompt:')
    print(message_content)
    print('*-*'*25)
# --------------------------------------
def print_response_tab(model_name,response,index):
    print(f'Response index {index}')
    print(model_name)
    print(response)
    print('*-*' * 25)
# --------------------------------------
def get_message_content():
    print('You:')
    content = input()
    return content
# --------------------------------------
def choose_model(models_list):
    print('Choose a model by index:')
    for i in range(len(models_list)):
        print(f'{i+1} : {models_list[i]}')
    index = int(input())
    return models_list[index-1]
# --------------------------------------
def get_response(curr_Chat:Chat_object,model:str):
    start = time.time()
    response = ollama.chat(model=model, messages=curr_Chat.messages)
    end = time.time()
    return response,(end-start)
# --------------------------------------
def get_chat_title():
    return input('Choose a chat title: ').strip()
# --------------------------------------
# Start Ollama server automatically and hide output
def start_ollama_server():
    # Run the ollama serve command in a background process and suppress output
    process = subprocess.Popen(
        ["ollama", "serve"], 
        stdout=subprocess.DEVNULL,  # Hide standard output
        stderr=subprocess.DEVNULL   # Hide standard error
    )
    time.sleep(1)  # Give some time for the server to start
    return process
# --------------------------------------
