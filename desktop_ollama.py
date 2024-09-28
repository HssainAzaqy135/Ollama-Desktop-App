from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty, NumericProperty
from kivy.core.window import Window
from kivy.clock import Clock
import ollama
import time
import subprocess
import os

# Set the window background color to a pastel blue
Window.clearcolor = (0.8, 0.9, 1, 1)  # Light pastel blue

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
    # Run the ollama serve command in a background process and suppress output
    process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,  # Hide standard output
    )
    time.sleep(3)  # Give some time for the server to start
    return process

def check_gpu_availability():
    """Check if GPUs are available using nvidia-smi."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            gpu_list = result.stdout.strip().split('\n')
            return gpu_list  # Returns GPU index and name
        else:
            print("GPU not detected. Make sure NVIDIA drivers are installed and running.")
            return []
    except FileNotFoundError:
        print("nvidia-smi not found. Please ensure CUDA is installed and accessible.")
        return []

def get_response(curr_chat: ChatObject, model: str, selected_gpu: str):
    start = time.time()

    # Set the selected GPU
    if selected_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu.split(":")[0]  # Extract the GPU index
        print(f"Running on GPU {selected_gpu}")
    else:
        print("No GPU selected, running on CPU.")

    response = ollama.chat(model=model, messages=curr_chat.messages)
    end = time.time()
    return response, (end - start)

class ScrollableLabel(ScrollView):
    text = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', size_hint_y=None)
        self.layout.bind(minimum_height=self.layout.setter('height'))
        self.label = Label(text=self.text, size_hint_y=None, markup=True, color=(0.2, 0.2, 0.2, 1))
        self.label.bind(texture_size=self._set_label_height, size=self._set_label_width)
        self.layout.add_widget(self.label)
        self.add_widget(self.layout)

    def _set_label_height(self, instance, size):
        self.label.height = size[1]
        self.layout.height = size[1]

    def _set_label_width(self, instance, size):
        self.label.text_size = (size[0], None)

    def update_text(self, new_text):
        self.label.text += new_text
        Clock.schedule_once(self._scroll_to_bottom, 0.1)

    def _scroll_to_bottom(self, dt):
        self.scroll_y = 0

class LlamaChatApp(App):
    response_window_height = NumericProperty(400)  # Default height, can be configured

    def build(self):
        self.title = "LLaMA Desktop App"

        # Initialize chat and available models
        self.curr_chat = ChatObject(title="Chat with LLaMA")
        self.available_models = get_available_models()
        self.selected_model = None

        # Get available GPUs
        self.gpus = check_gpu_availability()
        self.selected_gpu = None

        # Create a layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Create a prompt label and entry field
        prompt_label = Label(text="Enter your prompt:", color=(0.2, 0.2, 0.2, 1), size_hint_y=None, height=30)
        layout.add_widget(prompt_label)

        self.prompt_entry = TextInput(multiline=False, size_hint_y=None, height=40,
                                      background_color=(1, 1, 1, 1),
                                      foreground_color=(0.2, 0.2, 0.2, 1))
        layout.add_widget(self.prompt_entry)

        # Create a dropdown for model selection
        self.model_spinner = Spinner(
            text='Choose a model',
            values=self.available_models,
            size_hint_y=None,
            height=50,
            background_color=(0.6, 0.8, 1, 1),
            color=(0.2, 0.2, 0.2, 1)
        )
        layout.add_widget(self.model_spinner)

        # Create a dropdown for GPU selection
        self.gpu_spinner = Spinner(
            text='Select GPU (or leave blank for CPU)',
            values=self.gpus if self.gpus else ['No GPUs available'],
            size_hint_y=None,
            height=50,
            background_color=(0.6, 0.8, 1, 1),
            color=(0.2, 0.2, 0.2, 1)
        )
        layout.add_widget(self.gpu_spinner)

        # Create a button to generate the response
        generate_button = Button(text="Generate Response", size_hint_y=None, height=40,
                                 background_color=(0.6, 0.8, 1, 1), color=(0.2, 0.2, 0.2, 1))
        generate_button.bind(on_press=self.generate_response)
        layout.add_widget(generate_button)

        # Create a response label
        response_label = Label(text="Response:", color=(0.2, 0.2, 0.2, 1), size_hint_y=None, height=30)
        layout.add_widget(response_label)

        # Create a ScrollableLabel for the response output with configurable height
        self.response_output = ScrollableLabel(size_hint_y=None, height=self.response_window_height)
        layout.add_widget(self.response_output)

        return layout

    def generate_response(self, instance):
        self.selected_model = self.model_spinner.text
        self.selected_gpu = self.gpu_spinner.text

        if self.selected_model == 'Choose a model':
            self.show_error("Please choose a model first.")
            return

        prompt = self.prompt_entry.text
        if prompt:
            self.curr_chat.prompts.append(prompt)
            self.curr_chat.messages.append({"role": "user", "content": prompt})

            response, total_time = get_response(curr_chat=self.curr_chat, model=self.selected_model, selected_gpu=self.selected_gpu)

            # Display the response
            new_text = f"\n\nUser: {prompt}\n\nResponse: {response['message']['content']}\nComputation time: {total_time:.2f}s\n"
            self.response_output.update_text(new_text)
            self.curr_chat.replies.append(response['message']['content'])
            self.curr_chat.messages.append(response['message'])

            # Clear the prompt entry
            self.prompt_entry.text = ""
        else:
            self.show_error("Please enter a prompt.")

    def show_error(self, message):
        popup = Popup(title="Error",
                      content=Label(text=message, color=(0.2, 0.2, 0.2, 1)),
                      size_hint=(0.6, 0.4),
                      background_color=(0.8, 0.9, 1, 1))
        popup.open()

    def on_stop(self):
        # Stop the Ollama server when the app is closed
        if self.ollama_server:
            self.ollama_server.terminate()
            print("Ollama server has been shut down.")

if __name__ == "__main__":
    # Start the Ollama server and store the process handle
    app = LlamaChatApp()
    app.ollama_server = start_ollama_server()
    # Visual Configs
    app.response_window_height = 400
    app.run()
