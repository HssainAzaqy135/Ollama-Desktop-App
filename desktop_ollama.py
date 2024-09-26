import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
import ollama
import json
import time
import subprocess

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
        stderr=subprocess.DEVNULL  # Hide standard error
    )
    time.sleep(3)  # Give some time for the server to start
    return process


def get_response(curr_chat: ChatObject, model: str):
    start = time.time()
    response = ollama.chat(model=model, messages=curr_chat.messages)
    end = time.time()
    return response, (end - start)


class LlamaChatApp(App):
    def build(self):
        self.title = "LLaMA Desktop App"

        # Initialize chat and available models
        self.curr_chat = ChatObject(title="Chat with LLaMA")
        self.available_models = get_available_models()
        self.selected_model = None

        # Create a layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Create a prompt label and entry field
        prompt_label = Label(text="Enter your prompt:", color=(0.2, 0.4, 0.6, 1))
        layout.add_widget(prompt_label)

        self.prompt_entry = TextInput(multiline=False, size_hint_y=None, height=40,
                                      background_color=(1, 1, 1, 1),
                                      foreground_color=(0, 0, 0, 1))
        layout.add_widget(self.prompt_entry)

        # Create a dropdown for model selection
        self.model_spinner = Spinner(
            text='Choose a model',
            values=self.available_models,
            size_hint=(None, None),
            size=(200, 44),
            pos_hint={'center_x': 0.5},
            background_color=(0.6, 0.8, 1, 1),  # Light pastel blue background
            color=(0, 0, 0, 1)  # Black text
        )
        layout.add_widget(self.model_spinner)

        # Create a response label inside a ScrollView
        response_label = Label(text="Response:", color=(0.2, 0.4, 0.6, 1))
        layout.add_widget(response_label)

        # Create a ScrollView for the response output
        self.response_scroll = ScrollView(size_hint=(1, None), size=(400, 200))
        self.response_output = Label(text="", size_hint_y=None, color=(0, 0, 0, 1))
        self.response_output.bind(size=self.response_output.setter('text_size'))
        self.response_output.text_size = (self.response_scroll.width, None)  # Auto size based on width
        self.response_scroll.add_widget(self.response_output)

        layout.add_widget(self.response_scroll)

        # Create a button to generate the response
        generate_button = Button(text="Generate Response", size_hint_y=None, height=40)
        generate_button.bind(on_press=self.generate_response)
        layout.add_widget(generate_button)

        return layout

    def generate_response(self, instance):
        self.selected_model = self.model_spinner.text

        if self.selected_model == 'Choose a model':
            self.show_error("Please choose a model first.")
            return

        prompt = self.prompt_entry.text
        if prompt:
            self.curr_chat.prompts.append(prompt)
            self.curr_chat.messages.append({"role": "user", "content": prompt})

            response, total_time = get_response(curr_chat=self.curr_chat, model=self.selected_model)

            # Display the response
            self.response_output.text = f"{response['message']['content']}\nComputation time: {total_time:.2f}s"
            self.curr_chat.replies.append(response['message']['content'])
            self.curr_chat.messages.append(response['message'])
        else:
            self.show_error("Please enter a prompt.")

    def show_error(self, message):
        popup = Popup(title="Error", content=Label(text=message), size_hint=(0.6, 0.4),
                      background_color=(0.8, 0.9, 1, 1))
        popup.open()


if __name__ == "__main__":
    start_ollama_server()
    LlamaChatApp().run()
