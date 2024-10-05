import platform
import subprocess
import os
import psutil
import customtkinter as ctk
import ollama
import time

class ChatObject:
    def __init__(self, name: str):
        self.name = name
        self.messages = []
        self.reply_time = []
        self.addressed_models =[]


class CenteredInputDialog(ctk.CTkInputDialog):
    def __init__(self, master=None, width=300, height=200, **kwargs):
        super().__init__(master, **kwargs)
        self.width = width
        self.height = height
        self.withdraw()  # Hide the window initially
        self.after(0, self.center_and_show)  # Schedule centering and showing

    def center_and_show(self):
        self.update_idletasks()  # Ensure size calculations are correct
        # Force the desired size
        self.geometry(f"{self.width}x{self.height}")
        # Recalculate the position
        x = (self.winfo_screenwidth() // 2) - (self.width // 2)
        y = (self.winfo_screenheight() // 2) - (self.height // 2)
        # Set the geometry with both size and position
        self.geometry(f"{self.width}x{self.height}+{x}+{y}")
        self.deiconify()  # Show the window


def get_available_models():
    models_dict = ollama.list()
    model_list = [model['name'] for model in models_dict['models']]
    return model_list

def start_ollama_server():
    process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(3)  # Give some time for the server to start
    return process

def terminate_with_children(process):
    parent = psutil.Process(process.pid)
    for child in parent.children(recursive=True):  # Terminate child processes
        child.terminate()
    parent.terminate()
    parent.wait()

def check_gpu_availability():
    try:
        gpu_list = ["CPU"]  # Start with CPU option
        if platform.system() == "Darwin":
            from torch.backends import mps
            if mps.is_available():
                gpu_list.append("MPS")
        else:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                gpus = result.stdout.strip().split('\n')
                gpu_list.extend(gpus)
        return gpu_list
    except FileNotFoundError:
        print("nvidia-smi not found. Please ensure CUDA or MPS is installed and accessible.")
        return ["CPU"]

def get_response(curr_chat: ChatObject, model: str, selected_gpu: str):
    start = time.time()
    if selected_gpu != "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu.split(":")[0]
        print(f"Running on GPU {selected_gpu}")
    else:
        print("No GPU selected, running on CPU.")
    response = ollama.chat(model=model, messages=curr_chat.messages)
    end = time.time()
    return response, (end - start)


def validate_data_structure(chat_data):
    try:
        if not isinstance(chat_data["messages"], list):
            raise ValueError("Invalid chat data format: 'messages' should be a list")
        if not isinstance(chat_data["reply_times"], list):
            raise ValueError("Invalid chat data format: 'reply_times' should be a list")
        if not isinstance(chat_data["addressed_models"], list):
            raise ValueError("Invalid chat data format: 'addressed_models' should be a list")

        # Validate the structure of each message in the 'messages' list
        for index, message in enumerate(chat_data["messages"]):
            if not isinstance(message, dict):
                raise ValueError(f"Invalid message format at index {index}: Expected a dictionary")
            if not all(key in message for key in ["role", "content"]):
                raise ValueError(f"Invalid message format at index {index}: Missing 'role' or 'content' keys")
            if not message["role"] or not message["content"]:
                raise ValueError(f"Invalid message format at index {index}: 'role' or 'content' cannot be empty")

        # Validate that the data aligns
        num_of_reply_times = len(chat_data["reply_times"])
        num_of_addresed_models = len(chat_data["addressed_models"])
        if(len(chat_data["reply_times"]) != len(chat_data["addressed_models"])):
            raise ValueError(f"Number of  reply_times {num_of_reply_times} doesn't match number of addressed_models {num_of_addresed_models}")

    except ValueError as e:
        return e

    return None #Meaning passed check

