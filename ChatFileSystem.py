import sqlite3
import json
from utils import ChatObject

class ChatMemory:
    def __init__(self, db_path='chats.db'):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                messages TEXT,
                reply_times TEXT,
                addressed_models TEXT,
                instructions TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def add_chat(self, chat_object: ChatObject):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO chats (name, messages, reply_times, addressed_models, instructions)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            chat_object.name,
            json.dumps(chat_object.messages),
            json.dumps(chat_object.reply_times),
            json.dumps(chat_object.addressed_models),
            chat_object.instructions
        ))

        conn.commit()
        conn.close()

    def get_chat_by_name(self, name: str) -> ChatObject:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM chats WHERE name = ?', (name,))
        chat_data = cursor.fetchone()
        conn.close()

        if chat_data:
            return self._row_to_chat_object(chat_data)
        return None

    def update_chat(self, chat_object: ChatObject):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE chats
            SET messages = ?, reply_times = ?, addressed_models = ?, instructions = ?
            WHERE name = ?
        ''', (
            json.dumps(chat_object.messages),
            json.dumps(chat_object.reply_times),
            json.dumps(chat_object.addressed_models),
            chat_object.instructions,
            chat_object.name
        ))

        conn.commit()
        conn.close()

    def delete_chat_by_name(self, name: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM chats WHERE name = ?', (name,))
        conn.commit()
        conn.close()

    def list_chat_names(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT name FROM chats')
        chat_names = [row[0] for row in cursor.fetchall()]
        conn.close()

        return chat_names

    def _row_to_chat_object(self, row):
        return ChatObject(
            name=row[1],
            messages=json.loads(row[2]),
            reply_times=json.loads(row[3]),
            addressed_models=json.loads(row[4]),
            instructions=row[5]
        )

    def clear_all_chats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chats')
        conn.commit()
        conn.close()

    def reset_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS chats')
        self._initialize_database()
        conn.commit()
        conn.close()