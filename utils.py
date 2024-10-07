import platform
import subprocess
import os
import psutil
import customtkinter as ctk
import ollama
import time

class ChatObject:
    def __init__(self, name: str,
                 messages: list = None,
                 reply_times: list = None,
                 addressed_models: list = None,
                 instructions: str = None):
        self.name = name
        self.messages = messages if messages is not None else []
        self.reply_times = reply_times if reply_times is not None else []
        self.addressed_models = addressed_models if addressed_models is not None else []
        self.instructions = instructions if instructions is not None else ""
class CenteredTextInputDialog(ctk.CTkToplevel):
    def __init__(self, master=None, width=300, height=200, max_length=None, initial_text="", **kwargs):
        super().__init__(master)
        self.width = width
        self.height = height
        self.max_length = max_length  # Max length for input text
        self.title(kwargs.get("title", "Input"))
        self.result = None

        self.frame = ctk.CTkFrame(self)
        self.frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.label = ctk.CTkLabel(self.frame, text=kwargs.get("text", "Enter text:"))
        self.label.pack(pady=(0, 10))

        self.text_widget = ctk.CTkTextbox(self.frame, width=width-40, height=height-120)  # Adjusted height
        self.text_widget.pack(pady=(0, 10), fill="both", expand=True)

        # Insert initial text into the text widget
        self.text_widget.insert("1.0", initial_text)  # Insert at line 1, character 0

        if self.max_length is not None:
            self.text_widget.bind("<KeyPress>", self.prevent_excess_input)

        button_frame = ctk.CTkFrame(self.frame)
        button_frame.pack(fill="x", pady=(10, 0))  # Added vertical padding

        button_style = {
            "width": 120,  # Increased width
            "height": 40,  # Increased height
            "corner_radius": 8,  # Rounded corners
            "border_width": 2,  # Added border
            "font": ("Arial", 14, "bold")  # Larger, bold font
        }

        self.ok_button = ctk.CTkButton(button_frame, text="OK", command=self.on_ok, **button_style)
        self.ok_button.pack(side="left", padx=(0, 10))

        self.cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=self.on_cancel, **button_style)
        self.cancel_button.pack(side="right")

        self.withdraw()  # Hide the window initially
        self.after(0, self.center_and_show)  # Schedule centering and showing

    def prevent_excess_input(self, event=None):
        """Prevent further input when the text reaches max_length."""
        current_text = self.text_widget.get("1.0", "end-1c")  # Get current text without trailing newline
        if len(current_text) >= self.max_length and event.keysym not in ("BackSpace", "Delete"):
            return "break"  # Block any further input except for backspace and delete

    def center_and_show(self):
        self.update_idletasks()  # Ensure size calculations are correct
        x = (self.winfo_screenwidth() // 2) - (self.width // 2)
        y = (self.winfo_screenheight() // 2) - (self.height // 2)
        self.geometry(f"{self.width}x{self.height}+{x}+{y}")
        self.deiconify()  # Show the window

    def on_ok(self):
        self.result = self.text_widget.get("1.0", ctk.END).strip()
        self.destroy()  # Close the window

    def on_cancel(self):
        self.result = None
        self.destroy()  # Close the window

    def get_input(self):
        self.master.wait_window(self)
        return self.result


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

    # If instructions exist, prepend them to the messages
    messages = curr_chat.messages.copy()
    if curr_chat.instructions:
        messages.insert(0, {"role": "system", "content": curr_chat.instructions})
    response = ollama.chat(model=model, messages=messages)
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

