import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, scrolledtext
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
        self.title = title
        self.messages = []
        self.reply_time = []

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

class LlamaDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Llama Desktop App")

        # Set the window size
        window_width = 900
        window_height = 700

        # Get screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate the position to center the window
        position_x = int((screen_width / 2) - (window_width / 2))
        position_y = int((screen_height / 2) - (window_height / 2))

        # Set the geometry of the window with position and size
        self.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

        self.executor = ThreadPoolExecutor(max_workers=2)
        self.chats = []
        self.current_chat = None
        self.available_models = get_available_models()
        self.selected_model = None
        self.gpus = check_gpu_availability()
        self.selected_gpu = None

        self.text_color = {"red": 0.2, "green": 0.2, "blue": 0.2}

        self.create_widgets()
        self.new_chat()

    def create_widgets(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(1, weight=1)

        new_chat_button = ctk.CTkButton(self.sidebar, text="New Chat", command=self.new_chat)
        new_chat_button.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.chat_list = tk.Listbox(self.sidebar, bg='#2b2b2b', fg='white', selectbackground='#4a4a4a')
        self.chat_list.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="nsew")
        self.chat_list.bind('<<ListboxSelect>>', self.on_chat_select)

        # Notebook (tabbed interface)
        self.notebook = ctk.CTkTabview(self)
        self.notebook.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Create tabs
        self.chat_tab = self.notebook.add("Chat")
        self.settings_tab = self.notebook.add("Settings")

        self.create_chat_tab()
        self.create_settings_tab()

    def create_chat_tab(self):
        self.chat_tab.grid_columnconfigure(0, weight=1)
        self.chat_tab.grid_rowconfigure(4, weight=1)

        # Prompt entry
        prompt_label = ctk.CTkLabel(self.chat_tab, text="Enter your prompt:")
        prompt_label.grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.prompt_entry = ctk.CTkEntry(self.chat_tab, height=30)
        self.prompt_entry.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        # Create a frame for 2x2 grid (dropdowns and buttons)
        selection_frame = ctk.CTkFrame(self.chat_tab)
        selection_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        # Configure equal weight for rows and columns to ensure proportional sizing
        selection_frame.grid_columnconfigure((0, 1), weight=1, uniform="col")
        selection_frame.grid_rowconfigure((0, 1), weight=1, uniform="row")

        # Model dropdown in first row, first column
        self.model_var = tk.StringVar(value="Choose a model")
        self.model_dropdown = ctk.CTkOptionMenu(selection_frame, variable=self.model_var, values=self.available_models)
        self.model_dropdown.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 5))

        # GPU dropdown in first row, second column
        self.gpu_var = tk.StringVar(value="Select GPU (or leave blank for CPU)")
        self.gpu_dropdown = ctk.CTkOptionMenu(selection_frame, variable=self.gpu_var, values=self.gpus)
        self.gpu_dropdown.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=(0, 5))

        # "Generate Response" button in second row, first column
        generate_button = ctk.CTkButton(selection_frame, text="Generate Response", command=self.generate_response)
        generate_button.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=(5, 0))

        # "Clear Chat" button in second row, second column
        clear_button = ctk.CTkButton(selection_frame, text="Clear Chat", command=self.clear_chat)
        clear_button.grid(row=1, column=1, sticky="nsew", padx=(5, 0), pady=(5, 0))

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(self.chat_tab, wrap=tk.WORD, bg='#2b2b2b', fg='white')
        self.chat_display.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        self.chat_display.config(state=tk.DISABLED)

        # Access and configure the scrollbar
        for child in self.chat_display.winfo_children():
            if isinstance(child, tk.Scrollbar):
                child.config(bg='#1e1e1e', troughcolor='#1e1e1e')  # Change scrollbar and trough color
                break  # Stop after finding the scrollbar

    def create_settings_tab(self):
        self.settings_tab.grid_columnconfigure(0, weight=1)
        self.settings_tab.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(self.settings_tab, text="Text Color", font=("Arial", 16)).grid(row=0, column=0, pady=(0, 10))

        # Color sliders
        self.red_slider = self.create_color_slider(self.settings_tab, "Red", 1)
        self.green_slider = self.create_color_slider(self.settings_tab, "Green", 2)
        self.blue_slider = self.create_color_slider(self.settings_tab, "Blue", 3)

        # Apply button
        apply_button = ctk.CTkButton(self.settings_tab, text="Apply Changes", command=self.apply_color_changes)
        apply_button.grid(row=4, column=0, pady=(20, 0))

    def create_color_slider(self, parent, color, row):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text=f"{color}:").grid(row=0, column=0, padx=(0, 10))
        slider = ctk.CTkSlider(frame, from_=0, to=1, number_of_steps=100)
        slider.grid(row=0, column=1, sticky="ew")
        slider.set(self.text_color[color.lower()])

        return slider

    def apply_color_changes(self):
        self.text_color["red"] = self.red_slider.get()
        self.text_color["green"] = self.green_slider.get()
        self.text_color["blue"] = self.blue_slider.get()

        new_color = self.rgb_to_hex(self.text_color["red"], self.text_color["green"], self.text_color["blue"])
        self.chat_display.config(fg=new_color)  # Update the text color
        self.prompt_entry.configure(text_color=new_color)

        # Change chat_display background color
        background_color = '#1e1e1e'  # Example: dark background
        self.chat_display.config(bg=background_color)  # Update the background color

    def rgb_to_hex(self, r, g, b):
        return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'

    def new_chat(self):
        chat = ChatObject(f"Chat {len(self.chats) + 1}")
        self.chats.append(chat)
        self.chat_list.insert(tk.END, chat.title)
        self.chat_list.selection_clear(0, tk.END)
        self.chat_list.selection_set(tk.END)
        self.on_chat_select(None)

    def on_chat_select(self, event):
        selection = self.chat_list.curselection()
        if selection:
            index = selection[0]
            self.current_chat = self.chats[index]
            self.update_chat_display()

    def update_chat_display(self):
        # Clear and prepare the chat display
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)

        # Define the bold tag if it doesn't already exist
        self.chat_display.tag_configure("bold", font=("Arial", 12, "bold"))

        # Loop through messages and display them
        for i in range(len(self.current_chat.messages)):
            if self.current_chat.messages[i]['role'] != 'user':
                # Insert the header in bold
                self.chat_display.insert(tk.END, f"{self.selected_model}:\n ", "bold")
                # Insert the message content normally
                self.chat_display.insert(tk.END, f"{self.current_chat.messages[i]['content']}\n\n")

                self.chat_display.insert(tk.END, f"Response time: {self.current_chat.reply_time[int(i/2)]:.2f} seconds\n\n")
            else:
                self.chat_display.insert(tk.END, "User:\n ", "bold")
                # Insert the message content normally
                self.chat_display.insert(tk.END, f"{self.current_chat.messages[i]['content']}\n\n")



        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def clear_chat(self):
        if self.current_chat:
            self.current_chat.messages.clear()
            self.current_chat.reply_time.clear()
            self.update_chat_display()
            print("Chat cleared.")

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

        self.current_chat.messages.append({"role": "user", "content": prompt})
        self.update_chat_display()

        self.executor.submit(self.fetch_response_async, prompt)

    def fetch_response_async(self, prompt):
        response, time_taken = get_response(self.current_chat, self.selected_model, self.selected_gpu)
        self.after(0, self.update_ui_with_response, response, time_taken)

    def update_ui_with_response(self, response, time_taken):
        self.current_chat.messages.append({"role": "assistant", "content": response['message']['content']})
        self.current_chat.reply_time.append(time_taken)

        self.update_chat_display()
        self.chat_display.see(tk.END)

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
    app = LlamaDesktopApp()
    app.ollama_server = start_ollama_server()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()