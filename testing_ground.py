import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, scrolledtext,filedialog
import json
from utils import *
import ollama
from ChatFileSystem import ChatMemory  # Add this import
from concurrent.futures import ThreadPoolExecutor

ctk.set_appearance_mode("dark") # We don't  believe in light mode
ctk.set_default_color_theme("blue")

class LlamaDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.ollama_server = start_ollama_server()
        self.title("Llama Desktop App")
        # Set the window size
        self.window_width = 900
        self.window_height = 700
        self.font_size = 12 #Default font size
        self.slider_value_vars = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.chat_memory = ChatMemory()  # Initialize ChatMemory
        self.chat_keys = self.chat_memory.list_chat_ids()
        self.current_chat = None
        self.available_models = get_available_models()
        self.selected_model = None
        self.gpus = check_gpu_availability()
        self.selected_gpu = None
        self.create_widgets()
        self.load_chats_from_memory()  # Load existing chats from memory
        self.config_window_geometry()

    def config_window_geometry(self):
        # Get screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate the position to center the window
        position_x = int((screen_width / 2) - (self.window_width / 2))
        position_y = int((screen_height / 2) - (self.window_height / 2))

        # Set the geometry of the window with position and size
        self.geometry(f"{self.window_width}x{self.window_height}+{position_x}+{position_y}")

        self.text_color = {"red": 0.2, "green": 0.8, "blue": 0.2}


        self.red_slider.set(self.text_color['red'])
        self.green_slider.set(self.text_color['green'])
        self.blue_slider.set(self.text_color['blue'])
        self.apply_color_changes()
        self.font_size_slider.set(self.font_size)
        self.apply_font_size_changes()

    def create_widgets(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(1, weight=1)
        # New Chat button
        new_chat_button = ctk.CTkButton(self.sidebar, text="New Chat", command=self.prompt_new_chat)
        new_chat_button.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.chat_list = tk.Listbox(self.sidebar, bg='#2b2b2b', fg='white', selectbackground='#4a4a4a')
        self.chat_list.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="nsew")
        self.chat_list.bind('<<ListboxSelect>>', self.on_chat_select)

        # Add Remove Chat button
        remove_chat_button = ctk.CTkButton(self.sidebar, text="Remove Chat", command=self.remove_chat)
        remove_chat_button.grid(row=2, column=0, padx=20, pady=(10, 20))

        # Notebook (tabbed interface)
        self.notebook = ctk.CTkTabview(self)
        self.notebook.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Create tabs
        self.chat_tab = self.notebook.add("Chat")
        self.settings_tab = self.notebook.add("Settings")

        # Add Save Chat button
        save_chat_button = ctk.CTkButton(self.sidebar, text="Save Chat", command=self.save_chat)
        save_chat_button.grid(row=3, column=0, padx=20, pady=(10, 20))

        # Add Import Chat button
        import_chat_button = ctk.CTkButton(self.sidebar, text="Import Chat", command=self.import_chat)
        import_chat_button.grid(row=4, column=0, padx=20, pady=(10, 20))

        # Add Set Instructions button
        set_instructions_button = ctk.CTkButton(self.sidebar, text="Set Instructions", command=self.set_instructions)
        set_instructions_button.grid(row=5, column=0, padx=20, pady=(10, 20))

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
        self.settings_tab.grid_columnconfigure(1, weight=1)  # Add weight to column 1
        self.settings_tab.grid_rowconfigure(6, weight=1)  # Increased to accommodate new elements

        ctk.CTkLabel(self.settings_tab, text="Text Color", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Color sliders
        self.red_slider, red_value_label = self.create_color_slider(self.settings_tab, "Red", 1)
        self.green_slider, green_value_label = self.create_color_slider(self.settings_tab, "Green", 2)
        self.blue_slider, blue_value_label = self.create_color_slider(self.settings_tab, "Blue", 3)

        # Font size slider
        ctk.CTkLabel(self.settings_tab, text="Font Size", font=("Arial", 16)).grid(row=4, column=0, columnspan=2, pady=(20, 5))
        self.font_size_slider, font_size_value_label = self.create_slider(self.settings_tab, "Font Size", 5, 12, 24, 12)

        # Apply button
        apply_button = ctk.CTkButton(self.settings_tab, text="Apply Changes", command=self.apply_changes)
        apply_button.grid(row=6, column=0, columnspan=2, pady=(20, 0))

    def create_slider(self, parent, name, row, from_, to, number_of_steps):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text=f"{name}:").grid(row=0, column=0, padx=(0, 10))
        slider = ctk.CTkSlider(frame, from_=from_, to=to, number_of_steps=number_of_steps)
        slider.grid(row=0, column=1, sticky="ew")

        value_var = tk.StringVar()
        self.slider_value_vars[name] = value_var
        value_label = ctk.CTkLabel(frame, textvariable=value_var)
        value_label.grid(row=0, column=2, padx=(10, 0))

        slider.configure(command=lambda value: self.update_slider_value(name, value))
        slider.set((from_ + to) / 2)  # Set to middle value
        self.update_slider_value(name, slider.get())

        return slider, value_label

    def create_color_slider(self, parent, color, row):
        return self.create_slider(parent, color, row, 0, 1, 100)

    def update_slider_value(self, name, value):
        if name in ["Red", "Green", "Blue"]:
            self.slider_value_vars[name].set(f"{value:.2f}")
        else:
            self.slider_value_vars[name].set(f"{int(value)}")

    def load_chats_from_memory(self):
        chat_names = self.chat_memory.list_chat_names()
        for name in chat_names:
            self.chat_list.insert(tk.END,name)
    def prompt_new_chat(self):
        dialog = CenteredTextInputDialog(text="Enter a name for the new chat:",
                                     title="New Chat",
                                     height=250,
                                     width=350,
                                     max_length=50)
        name = dialog.get_input()
        if name is None:
            # Dialog was cancelled
            print("New chat creation cancelled")
            # You can add any other actions you want to take when cancelled
            return  # Exit the method without creating a new chat
        if name:
            self.new_chat(name)
        else:
            self.new_chat(f"New Chat")
    def new_chat(self, name, messages=None, reply_times=None,addressed_models = None):
        chat = ChatObject(name = name,
                          messages=messages,
                          reply_times=reply_times,
                          addressed_models= addressed_models)
        self.chat_keys.append(chat.creation_time)
        self.chat_list.insert(tk.END, chat.name)
        self.chat_list.selection_clear(0, tk.END)
        self.chat_list.selection_set(tk.END)
        self.current_chat = chat
        self.update_chat_display()
        self.chat_memory.add_chat(chat)  # Adding the new chat to memory

    def remove_chat(self):
        selection = self.chat_list.curselection()
        if selection:
            index = selection[0]
            removed_chat = self.chat_keys.pop(index)
            self.chat_list.delete(index)
            self.chat_memory.delete_chat_by_timestamp(removed_chat.creation_time)  # Remove chat from memory by timestamp

            # Update listbox items
            self.chat_list.delete(0, tk.END)
            for chat in self.chat_keys:
                self.chat_list.insert(tk.END, chat.name)

            # Select the next chat, or the last one if we removed the last chat
            if self.chat_keys:
                new_index = min(index, len(self.chat_keys) - 1)
                self.chat_list.selection_set(new_index)
                self.current_chat = self.chat_keys[new_index]
            else:
                self.current_chat = None

            self.update_chat_display()
            print(f"Removed chat: {removed_chat.name}")
        else:
            messagebox.showinfo("Info", "Please select a chat to remove.")

    def apply_changes(self):
        self.apply_color_changes()
        self.apply_font_size_changes()

    def apply_font_size_changes(self):
        self.font_size = int(self.font_size_slider.get())
        font = ("Arial", self.font_size)

        # Update font size for chat display
        self.chat_display.config(font=font)

        # Update font size for prompt entry
        self.prompt_entry.configure(font=font)

        # Update font size for chat list
        self.chat_list.config(font=font)

        # Update font size for model and GPU dropdowns
        self.model_dropdown.configure(font=font)
        self.gpu_dropdown.configure(font=font)

        # Update font size for all buttons and labels
        self.update_font_recursive(self, font)

        # Update the bold tag with the new font size (Scaled up by 2)
        self.chat_display.tag_configure("bold", font=("Arial", self.font_size + 2, "bold"))
        # Update the color sliders values
        self.update_slider_value("Font Size", self.font_size_slider.get())

    def update_font_recursive(self, widget, font):
        if isinstance(widget, (ctk.CTkButton, ctk.CTkLabel, ctk.CTkEntry, ctk.CTkOptionMenu)):
            widget.configure(font=font)

        for child in widget.winfo_children():
            self.update_font_recursive(child, font)

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
        # Update slider labels
        self.update_slider_value("Red", self.red_slider.get())
        self.update_slider_value("Green", self.green_slider.get())
        self.update_slider_value("Blue", self.blue_slider.get())

    def rgb_to_hex(self, r, g, b):
        return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'

    def on_chat_select(self, event):
        selection = self.chat_list.curselection()
        if selection:
            index = selection[0]
            self.current_chat = self.chat_memory.get_chat_by_timestamp(self.chat_keys[index])
            self.update_chat_display()

    def set_instructions(self):
        if not self.current_chat:
            messagebox.showerror("Error", "No chat selected. Please select or create a chat first.")
            return

        # Pass the current instructions as the initial text
        initial_text = self.current_chat.instructions
        dialog = CenteredTextInputDialog(self, text="Enter instructions for the current chat:",
                                         title="Set Instructions",
                                         height=300,
                                         width=500,
                                         max_length=1000,
                                         initial_text=initial_text)  # Pass the initial text
        # Wait for the dialog to be created
        self.wait_window(dialog)
        # Check if the dialog was cancelled
        if dialog.result is None:
            return
        # If the dialog wasn't cancelled, update the instructions
        if dialog.result:
            self.current_chat.instructions = dialog.result
            messagebox.showinfo("Success", "Instructions have been updated.")

    def update_chat_display(self):
        # Clear and prepare the chat display
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        # Define the bold tag if it doesn't already exist
        self.chat_display.tag_configure("bold", font=("Arial", self.font_size + 2, "bold"))

        if self.current_chat:
            self.chat_display.insert(tk.END, f"Current Chat: {self.current_chat.name} \n\n", "bold")
            # Loop through messages and display them
            for i in range(len(self.current_chat.messages)):
                if self.current_chat.messages[i]['role'] != 'user':
                    # Insert the header in bold
                    self.chat_display.insert(tk.END, f"{self.current_chat.addressed_models[i//2]}:\n ", "bold")
                    # Insert the message content normally
                    self.chat_display.insert(tk.END, f"{self.current_chat.messages[i]['content']}\n\n")

                    if i // 2 < len(self.current_chat.reply_times):
                        self.chat_display.insert(tk.END,
                                                 f"Response time: {self.current_chat.reply_times[i // 2]:.2f} seconds\n\n")
                else:
                    self.chat_display.insert(tk.END, "User:\n ", "bold")
                    # Insert the message content normally
                    self.chat_display.insert(tk.END, f"{self.current_chat.messages[i]['content']}\n\n")
        else:
            self.chat_display.insert(tk.END, "No chat selected. Create a new chat or select an existing one.")

        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def clear_chat(self):
        if self.current_chat:
            self.current_chat.messages.clear()
            self.current_chat.reply_times.clear()
            self.current_chat.addressed_models.clear()
            # Updating memory
            self.chat_memory.update_chat(self.current_chat)
            # Updating display
            self.update_chat_display()
            print("Chat cleared.(Didn't clear instructions)")

    def save_chat(self):
        if not self.current_chat:
            messagebox.showerror("Error", "No chat selected. Please select a chat to save.")
            return

        # Prepare chat data
        chat_data = {
            "name": self.current_chat.name,
            "messages": self.current_chat.messages,
            "reply_times": self.current_chat.reply_times,
            "addressed_models": self.current_chat.addressed_models,
            "instructions": self.current_chat.instructions  # Add instructions to saved data
        }

        # Open file dialog to choose save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"{self.current_chat.name}.json"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(chat_data, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Success", f"Chat saved successfully to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save chat: {str(e)}")

    def import_chat(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Select a chat file to import"
        )
        if file_path:
            try:
                # Attempt to load and parse the JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)

                # Check if chat_data is a dictionary
                if not isinstance(chat_data, dict):
                    raise ValueError("Invalid chat data format: expected a dictionary")

                # Validate that 'name', 'messages', 'reply_times', 'addressed_models', and 'instructions' are present
                if not all(key in chat_data for key in ["name", "messages", "reply_times", "addressed_models", "instructions"]):
                    raise ValueError("Invalid chat data format: Missing some required keys")

                # Validate chat_data types
                result = validate_data_structure(chat_data)
                if result:
                    raise result

            except json.JSONDecodeError:
                messagebox.showerror("Error", "Invalid JSON file. Please ensure the file is properly formatted.")
                return
            except ValueError as e:
                messagebox.showerror("Error", str(e))
                return
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import chat: {str(e)}")
                return

            # Only create a new chat if no exceptions occurred
            new_chat = ChatObject(chat_data["name"])
            new_chat.messages = chat_data["messages"]
            new_chat.reply_times = chat_data["reply_times"]
            new_chat.addressed_models = chat_data["addressed_models"]
            new_chat.instructions = chat_data["instructions"]

            # Add the new chat to the list and update the UI
            self.chat_keys.append(new_chat)
            self.chat_list.insert(tk.END, new_chat.name)
            self.chat_list.selection_clear(0, tk.END)
            self.chat_list.selection_set(tk.END)
            self.current_chat = new_chat
            self.update_chat_display()
            self.chat_memory.add_chat(new_chat)
            messagebox.showinfo("Success", f"Chat '{new_chat.name}' imported successfully")

    def generate_response(self):
        if not self.current_chat:
            messagebox.showerror("Error", "No chat selected.")
            return

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
        self.current_chat.reply_times.append(time_taken)
        self.current_chat.addressed_models.append(self.selected_model)
        self.chat_memory.update_chat(self.current_chat)  # Update chat in memory

        self.update_chat_display()
        self.chat_display.see(tk.END)

    def stop_ollama_server(self):
        if self.ollama_server:
            print("Stopping Ollama server...")
            # Terminate the process
            try:
                terminate_with_children(self.ollama_server)
                print("Ollama server stopped.")
            except Exception as e:
                print(f"Error terminating Ollama server process: {e}")

            self.ollama_server = None
    def on_closing(self):
        self.stop_ollama_server()
        self.destroy()

if __name__ == "__main__":
    app = LlamaDesktopApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()