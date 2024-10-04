from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.uix.scrollview import ScrollView
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.slider import Slider
from kivy.properties import StringProperty, NumericProperty, ListProperty
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
# Python modules
from concurrent.futures import ThreadPoolExecutor
import platform
import subprocess
import ollama
import time
import os
import psutil


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


class ScrollableLabel(ScrollView):
    text = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', size_hint_y=None)
        self.layout.bind(minimum_height=self.layout.setter('height'))
        self.label = Label(text=self.text, size_hint_y=None, markup=True)
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


class ColorSlider(BoxLayout):
    color_channel = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.label = Label(text=self.color_channel, size_hint_x=0.3)
        self.slider = Slider(min=0, max=1, value=0.5, size_hint_x=0.7)
        self.add_widget(self.label)
        self.add_widget(self.slider)


class SettingsTab(BoxLayout):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.orientation = 'vertical'
        self.spacing = 10
        self.padding = 10

        # Text Color Sliders
        self.add_widget(Label(text="Text Color", font_size=16))
        self.text_red = ColorSlider(color_channel='Red')
        self.text_green = ColorSlider(color_channel='Green')
        self.text_blue = ColorSlider(color_channel='Blue')
        self.add_widget(self.text_red)
        self.add_widget(self.text_green)
        self.add_widget(self.text_blue)

        # Apply Button
        self.apply_button = Button(
            text="Apply Changes",
            size_hint=(0.3, 0.3),  # 30% of the parent size for both width and height
            pos_hint={'center_x': 0.5, 'center_y': 0.5}  # Center the button
        )
        self.apply_button.bind(on_press=self.apply_changes)
        self.add_widget(self.apply_button)

        # Set initial values for text color
        self.text_red.slider.value = 0.2
        self.text_green.slider.value = 0.2
        self.text_blue.slider.value = 0.2

    def apply_changes(self, instance):
        text_color = (self.text_red.slider.value, self.text_green.slider.value, self.text_blue.slider.value, 1)
        self.app.update_colors(text_color)


class LlamaChatApp(App):
    executor = ThreadPoolExecutor(max_workers=2)
    response_window_height = NumericProperty(400)
    background_color = ListProperty([0.8, 0.9, 1, 1])  # Fixed background color
    text_color = ListProperty([0.2, 0.8, 0.2, 1])
    # Window properties
    Window.clearcolor = (0.0, 1, 0.2, 1)
    Window.size = Window.system_size
    # -----------------
    def build(self):
        self.title = "LLaMA Desktop App"

        # Initialize chat and available models
        self.curr_chat = ChatObject(title="Chat with LLaMA")
        self.available_models = get_available_models()
        self.selected_model = None

        # Get available GPUs
        self.gpus = check_gpu_availability()
        self.selected_gpu = None

        # Create the main layout
        main_layout = BoxLayout(orientation='vertical')
        with main_layout.canvas.before:
            self.bg_color = Color(*self.background_color)  # Set fixed background color
            self.bg_rect = Rectangle(pos=main_layout.pos, size=main_layout.size)
        main_layout.bind(size=self._update_rect, pos=self._update_rect)

        # Create the tabbed panel
        self.tab_panel = TabbedPanel()
        main_layout.add_widget(self.tab_panel)

        # Create the chat tab
        chat_tab = TabbedPanelItem(text='Chat')
        chat_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        chat_tab.content = chat_layout

        # Create the settings tab
        settings_tab = TabbedPanelItem(text='Settings')
        settings_tab.content = SettingsTab(app=self)

        # Add tabs to the panel
        self.tab_panel.add_widget(chat_tab)
        self.tab_panel.add_widget(settings_tab)
        self.tab_panel.default_tab = chat_tab

        # Add widgets to the chat layout
        prompt_label = Label(
            text="Enter your prompt:",
            size_hint_y=0.1,  # Relative height
            pos_hint={'center_x': 0.5}
        )
        chat_layout.add_widget(prompt_label)

        self.prompt_entry = TextInput(
            multiline=False,
            size_hint_y=0.1  # Relative height
        )
        chat_layout.add_widget(self.prompt_entry)

        # Create a GridLayout for spinners and buttons
        grid_layout = GridLayout(
            cols=2,
            spacing=5,
            size_hint_y=0.3  # Relative height for spinners and buttons together
        )

        # Add spinners and buttons to the GridLayout
        self.model_spinner = Spinner(
            text='Choose a model',
            values=self.available_models,
            size_hint=(0.5, 0.5)  # Take half the width and half the height of a grid cell
        )
        grid_layout.add_widget(self.model_spinner)

        self.gpu_spinner = Spinner(
            text='Select GPU (or leave blank for CPU)',
            values=self.gpus if self.gpus else ['No GPUs available'],
            size_hint=(0.5, 0.5)
        )
        grid_layout.add_widget(self.gpu_spinner)

        generate_button = Button(
            text="Generate Response",
            size_hint=(0.5, 0.5)  # Button takes half the width and half the height of a grid cell
        )
        generate_button.bind(on_press=self.generate_response)
        grid_layout.add_widget(generate_button)

        clear_button = Button(
            text="Clear Model Context",
            size_hint=(0.5, 0.5)
        )
        clear_button.bind(on_press=self.clear_context)
        grid_layout.add_widget(clear_button)

        # Add the grid layout to the main chat layout
        chat_layout.add_widget(grid_layout)

        response_label = Label(
            text="Response:",
            size_hint_y=0.05  # Smaller relative height for the label
        )
        chat_layout.add_widget(response_label)

        self.response_output = ScrollableLabel(
            size_hint_y=0.5  # Take up the remaining space
        )
        chat_layout.add_widget(self.response_output)

        return main_layout

    def clear_context(self, instance):
        # Clear the current chat context
        self.curr_chat.messages.clear()
        self.curr_chat.replies.clear()
        self.curr_chat.reply_time.clear()

        # Clear the response output
        self.response_output.label.text = ""  # Reset the response tab text
        print("Chat context and response cleared.")

    def _update_rect(self, instance, value):
        self.bg_rect.pos = instance.pos
        self.bg_rect.size = instance.size

    def update_colors(self, text_color):
        self.text_color = text_color
        self.apply_colors()

    def apply_colors(self):
        def update_widget_colors(widget):
            if isinstance(widget, Label) or isinstance(widget, Button) or isinstance(widget, Spinner):
                widget.color = self.text_color
            if isinstance(widget, TextInput):
                widget.foreground_color = self.text_color
            if isinstance(widget, TabbedPanel) or isinstance(widget, TabbedPanelItem):
                if hasattr(widget, 'content'):
                    update_widget_colors(widget.content)
            if hasattr(widget, 'children'):
                for child in widget.children:
                    update_widget_colors(child)

        update_widget_colors(self.tab_panel)

        # Update ScrollableLabel text color
        if hasattr(self, 'response_output'):
            self.response_output.label.color = self.text_color

    def generate_response(self, instance):
        self.selected_model = self.model_spinner.text
        self.selected_gpu = self.gpu_spinner.text

        if self.selected_model == 'Choose a model':
            self.show_error("Please choose a model first.")
            return

        prompt = self.prompt_entry.text
        if not prompt:
            self.show_error("Please enter a prompt.")
            return

        self.curr_chat.messages.append({"role": "user", "content": prompt})
        self.response_output.update_text(f"[b]User:[/b] {prompt}\n\n")

        # Schedule the background task using the thread pool executor
        self.executor.submit(self.fetch_response_async, prompt)

    def fetch_response_async(self, prompt):
        # This runs in the background
        response, time_taken = get_response(self.curr_chat, self.selected_model, self.selected_gpu)

        # Once the response is ready, schedule the UI update on the main thread
        Clock.schedule_once(lambda dt: self.update_ui_with_response(response, time_taken), 0)

    def update_ui_with_response(self, response, time_taken):
        # This runs on the main thread (UI thread)
        self.curr_chat.messages.append({"role": "assistant", "content": response['message']['content']})
        self.curr_chat.reply_time.append(time_taken)

        self.response_output.update_text(f"[b]{self.selected_model}:[/b] {response['message']['content']}\n\n")
        self.response_output.update_text(f"[i]Response time: {time_taken:.2f} seconds[/i]\n\n")

    def show_error(self, message):
        popup = Popup(title='Error',
                      content=Label(text=message),
                      size_hint=(0.6, 0.4))
        popup.open()

    def on_stop(self):
        # Terminate the Ollama server process when the app closes
        if hasattr(self, 'ollama_server') and self.ollama_server:
            print("Attempting to terminate the Ollama server...")

            # Gracefully ask the server to stop
            self.ollama_server.terminate()
            try:
                self.ollama_server.wait(timeout=2)  # Wait briefly for the process to exit
            except subprocess.TimeoutExpired:
                print("Ollama server is taking too long to terminate.")

            time.sleep(2)  # Additional wait to allow cleanup

            # Check if the process is still running
            if self.ollama_server.poll() is None:
                print("Ollama server did not terminate. Forcefully stopping it...")
                terminate_with_children(self.ollama_server)  # Terminate with child processes
            else:
                print("Ollama server stopped successfully.")



if __name__ == "__main__":
    # Start the Ollama server and store the process handle
    app = LlamaChatApp()
    app.ollama_server = start_ollama_server()
    # Visual Configs
    app.response_window_height = 300
    app.run()