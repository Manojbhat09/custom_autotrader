import streamlit as st
import bcrypt
import sqlite3
import jwt


class AuthManager:
    def __init__(self):
        self.conn = sqlite3.connect('users.db', check_same_thread=False)  # Disabling check_same_thread for simplicity
        self.c = self.conn.cursor()
        self.c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)''')
        self.conn.commit()
        
    def register(self, username, password):
        self.c.execute('''SELECT * FROM users WHERE username = ?''', (username,))
        if self.c.fetchone() is not None:
            st.sidebar.error("Username already exists.")
            return
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        self.c.execute('''INSERT INTO users (username, password) VALUES (?, ?)''', (username, hashed_password))
        self.conn.commit()
        st.sidebar.success("Registered successfully.")
        
    def login(self, username, password):
        self.c.execute('''SELECT * FROM users WHERE username = ?''', (username,))
        user = self.c.fetchone()
        if user is None:
            st.sidebar.error("Invalid username or password.")
            return False
        if bcrypt.checkpw(password.encode('utf-8'), user[1]):
            st.sidebar.success("Logged in successfully.")
            return True
        else:
            st.sidebar.error("Invalid username or password.")
            return False


def verify_token(token):
    try:
        # Decode the token
        decoded = jwt.decode(token, 'random_secret_key', algorithms=['HS256'])
        return decoded['user']
    except jwt.ExpiredSignatureError:
        st.error('Signature expired. Please log in again.')
        return None
    except jwt.InvalidTokenError:
        st.error('Invalid token. Please log in again.')
        return None