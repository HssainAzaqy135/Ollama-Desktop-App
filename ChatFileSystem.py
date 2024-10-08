import sqlite3
import json
from datetime import datetime
from utils import ChatObject  # Assuming ChatObject has already been updated as discussed.

class ChatMemory:
    def __init__(self, db_path='chats.db'):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                timestamp DATETIME PRIMARY KEY,  -- Use timestamp as the primary key
                name TEXT NOT NULL,
                messages TEXT,
                reply_times TEXT,
                addressed_models TEXT,
                instructions TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_chat(self, chat_object: ChatObject):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO chats (timestamp, name, messages, reply_times, addressed_models, instructions)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            chat_object.creation_time,  # Use the timestamp as the unique identifier
            chat_object.name,
            json.dumps(chat_object.messages),
            json.dumps(chat_object.reply_times),
            json.dumps(chat_object.addressed_models),
            chat_object.instructions
        ))

        conn.commit()
        conn.close()

    def get_chat_by_timestamp(self, timestamp: str) -> ChatObject:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM chats WHERE timestamp = ?', (timestamp,))
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
            WHERE timestamp = ?
        ''', (
            json.dumps(chat_object.messages),
            json.dumps(chat_object.reply_times),
            json.dumps(chat_object.addressed_models),
            chat_object.instructions,
            chat_object.creation_time  # Use the timestamp to identify which chat to update
        ))

        conn.commit()
        conn.close()

    def delete_chat_by_timestamp(self, timestamp: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM chats WHERE timestamp = ?', (timestamp,))
        conn.commit()
        conn.close()

    def list_chat_names(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT name FROM chats')
        chat_names = [row[0] for row in cursor.fetchall()] # Acceses names from names column, row[0] for getting the value
        conn.close()

        return chat_names

    def list_chat_ids(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT timestamp FROM chats')
        chat_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        return chat_ids

    def _row_to_chat_object(self, row):
        return ChatObject(
            name=row[1],
            messages=json.loads(row[2]),
            reply_times=json.loads(row[3]),
            addressed_models=json.loads(row[4]),
            instructions=row[5],
            creation_time=row[0]  # Use timestamp as creation time
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
