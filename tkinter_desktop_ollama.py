import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import platform
import subprocess
import ollama
import time
import os
import psutil
from concurrent.futures import ThreadPoolExecutor

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ChatObject:
    def __init__(self, title: str):
        self.prompts = []
        self.replies = []
        self.reply_time = []
        self.title = title
        self.messages = []


def get_available_models():
    models_dict = ollama.list()
    model_list = [model['name'] for model in models_dict['models']]
    return model_list


def start_ollama_server():
    process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
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

        # Check if the platform is macOS
        if platform.system() == "Darwin":
            # Assuming MPS is available on macOS
            from torch.backends import mps
            if mps.is_available():
                gpu_list.append("MPS")
            else:
                print("MPS is not available. Ensure you have a supported Apple GPU and PyTorch with MPS support.")
        else:
            # For other platforms, check for NVIDIA GPUs using nvidia-smi
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                gpus = result.stdout.strip().split('\n')
                gpu_list.extend(gpus)
            else:
                print("No NVIDIA GPU detected.")

        return gpu_list
    except FileNotFoundError:
        print("nvidia-smi not found. Please ensure CUDA or MPS is installed and accessible.")
        return ["CPU"]


def get_response(curr_chat: ChatObject, model: str, selected_gpu: str):
    start = time.time()

    if selected_gpu != "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu.split(":")[0]  # Extract the GPU index
        print(f"Running on GPU {selected_gpu}")
    else:
        print("No GPU selected, running on CPU.")

    response = ollama.chat(model=model, messages=curr_chat.messages)
    end = time.time()
    return response, (end - start)


class LlamaChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("LLaMA Desktop App")
        self.geometry("800x600")

        self.executor = ThreadPoolExecutor(max_workers=2)
        self.curr_chat = ChatObject(title="Chat with LLaMA")
        self.available_models = get_available_models()
        self.selected_model = None
        self.gpus = check_gpu_availability()
        self.selected_gpu = None

        self.create_widgets()

    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)

        # Prompt entry
        prompt_label = ctk.CTkLabel(main_frame, text="Enter your prompt:")
        prompt_label.grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.prompt_entry = ctk.CTkEntry(main_frame, height=30)
        self.prompt_entry.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        # Model and GPU selection
        selection_frame = ctk.CTkFrame(main_frame)
        selection_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        selection_frame.grid_columnconfigure((0, 1), weight=1)

        self.model_var = tk.StringVar(value="Choose a model")
        self.model_dropdown = ctk.CTkOptionMenu(selection_frame, variable=self.model_var, values=self.available_models)
        self.model_dropdown.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.gpu_var = tk.StringVar(value="Select GPU (or leave blank for CPU)")
        self.gpu_dropdown = ctk.CTkOptionMenu(selection_frame, variable=self.gpu_var, values=self.gpus)
        self.gpu_dropdown.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # Buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        button_frame.grid_columnconfigure((0, 1), weight=1)

        generate_button = ctk.CTkButton(button_frame, text="Generate Response", command=self.generate_response)
        generate_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        clear_button = ctk.CTkButton(button_frame, text="Clear Model Context", command=self.clear_context)
        clear_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # Response output
        response_label = ctk.CTkLabel(main_frame, text="Response:")
        response_label.grid(row=4, column=0, sticky="w", pady=(10, 5))

        self.response_output = ctk.CTkTextbox(main_frame, wrap="word")
        self.response_output.grid(row=5, column=0, sticky="nsew")
        self.response_output.configure(state="disabled")

    def clear_context(self):
        self.curr_chat.messages.clear()
        self.curr_chat.replies.clear()
        self.curr_chat.reply_time.clear()
        self.response_output.configure(state="normal")
        self.response_output.delete("1.0", tk.END)
        self.response_output.configure(state="disabled")
        print("Chat context and response cleared.")

    def generate_response(self):
        self.selected_model = self.model_var.get()
        self.selected_gpu = self.gpu_var.get()

        if self.selected_model == 'Choose a model':
            messagebox.showerror("Error", "Please choose a model first.")
            return

        prompt = self.prompt_entry.get()
        if not prompt:
            messagebox.showerror("Error", "Please enter a prompt.")
            return

        self.curr_chat.messages.append({"role": "user", "content": prompt})
        self.update_response_output(f"User: {prompt}\n\n")

        # Schedule the background task using the thread pool executor
        self.executor.submit(self.fetch_response_async, prompt)

    def fetch_response_async(self, prompt):
        response, time_taken = get_response(self.curr_chat, self.selected_model, self.selected_gpu)
        self.after(0, self.update_ui_with_response, response, time_taken)

    def update_ui_with_response(self, response, time_taken):
        self.curr_chat.messages.append({"role": "assistant", "content": response['message']['content']})
        self.curr_chat.reply_time.append(time_taken)

        self.update_response_output(f"{self.selected_model}: {response['message']['content']}\n\n")
        self.update_response_output(f"Response time: {time_taken:.2f} seconds\n\n")

    def update_response_output(self, text):
        self.response_output.configure(state="normal")
        self.response_output.insert(tk.END, text)
        self.response_output.see(tk.END)
        self.response_output.configure(state="disabled")

    def on_closing(self):
        if hasattr(self, 'ollama_server') and self.ollama_server:
            print("Attempting to terminate the Ollama server...")
            self.ollama_server.terminate()
            try:
                self.ollama_server.wait(timeout=2)
            except subprocess.TimeoutExpired:
                print("Ollama server is taking too long to terminate.")

            time.sleep(2)

            if self.ollama_server.poll() is None:
                print("Ollama server did not terminate. Forcefully stopping it...")
                terminate_with_children(self.ollama_server)
            else:
                print("Ollama server stopped successfully.")

        self.destroy()


if __name__ == "__main__":
    app = LlamaChatApp()
    app.ollama_server = start_ollama_server()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()