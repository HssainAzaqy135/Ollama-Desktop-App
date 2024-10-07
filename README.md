# { Ollama Desktop App }
 Python code for Ollama desktop app
By: 
* Husain Azaqy
* Maroun Eilabuni

- Notes: 
1. `The refered app is llama_desktop_app.py`
2. Need to install ollama to serve the backend
3. Dependencies:
    ```
   import customtkinter
   import ollama
   import psutil
   import json
    ```
## App Base Features: ($ marks done)
1. Allow prompting and getting response from LLama3 models with chat UI $
2. Model Selection $
3. Chat history and context switching $
4. Chat memory $
5. Gpu acceleration and gpu selection $
- 5.1. Windows $
- 5.2. Mac $

## Future features to be added: ($ marks done)
1. Instructions tab set per chat 
2. import and export chats to json format (maybe csv format too)
- 2.1 Json $
- 2.2 CSV
3. Special council chat, build model council. Analyser, responder, refiner and finalizer
4. Text to speach for responses
5. Enable code running and inspection for models
