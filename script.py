from flask import Flask, render_template, Response,request, jsonify,json
import cv2
from datetime import datetime, timedelta
import subprocess
import time
import os
import sqlite3
from flask_socketio import SocketIO, emit
from threading import Lock,Thread
import base64


lock= Lock()
# running other file using run()

app = Flask(__name__)
socketio = SocketIO(app)
# Connect to SQLite database
conn = sqlite3.connect('user_data.db', check_same_thread=False)
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    clear_photo TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS logins (
    user_id INTEGER,
    intime TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
''')


FRAMES_DIR = './data/video'
FPS = 5  # Frames per second
MAX_DURATION = 120 # Maximum duration for frame retention in seconds (1 hour)

def delete_old_frames():
    now = time.time()
    for filename in os.listdir(FRAMES_DIR):
        frame_path = os.path.join(FRAMES_DIR, filename)
        if os.path.isfile(frame_path):
            if now - os.path.getctime(frame_path) > MAX_DURATION:
                os.remove(frame_path)

# Function to merge users
def merge_users(primary_user_id, secondary_user_ids):
    # Update photo and logins for primary user
    # Delete secondary users
    pass

# Function to add a new user
def add_user(name, clear_photo):
    cursor.execute('INSERT INTO users (name, clear_photo) VALUES (?, ?)', (name, clear_photo))
    conn.commit()

# Function to update user information
def update_user(id, name, clear_photo):
    cursor.execute('UPDATE users SET name=?, clear_photo=? WHERE id=?', (name, clear_photo, id))
    conn.commit()

# Function to delete a user
def delete_user(id):
    cursor.execute('DELETE FROM users WHERE id=?', (id,))
    conn.commit()

# Function to retrieve user information by ID
def get_user(id):
    cursor.execute('SELECT * FROM users WHERE id=?', (id,))
    return cursor.fetchone()

# Function to add a login time
def add_login_time(user_id, intime):
    cursor.execute('INSERT INTO logins (user_id, intime) VALUES (?, ?)', (user_id, intime))
    conn.commit()


# add_user('John Doe', b'photo_data')
add_login_time(2, '2024-04-26 07:00:00')
# print(get_last_10_logins(1))

# Function to read frames from a folder and emit to clients
def emit_frames():
    frame_folder = FRAMES_DIR  # Replace with the path to your frame folder
    while True:
        for filename in os.listdir(frame_folder):
            time.sleep(1/FPS)
            if filename.endswith('.jpg'):
                filepath = os.path.join(frame_folder, filename)
                with open(filepath, 'rb') as f:
                    frame_data = f.read()
                    frame_encoded = base64.b64encode(frame_data)
                    frame_encoded = frame_encoded.decode('utf-8')
                    socketio.emit('video_frame', {'frame': frame_encoded}, namespace='/')

@socketio.on('connect', namespace='/')
def connect():
    t = Thread(target=emit_frames)
    t.daemon = True
    t.start()
    print('Client connected')

@socketio.on('disconnect', namespace='/')
def disconnect():
    print('Client disconnected')
    

@app.route('/')
def index():
    return render_template('index.html')

def generate_video():
    while True:
        delete_old_frames()
        frame_files = os.listdir(FRAMES_DIR)
        if frame_files:
            for filename in frame_files:
                frame_path = os.path.join(FRAMES_DIR, filename)
                frame = cv2.imread(frame_path)
                frame_encoded = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_encoded + b'\r\n')
            time.sleep(1/FPS)
        else:
            time.sleep(1)

@app.route('/video_feed')
def video_feed():
    # Start emitting frames in a separate thread
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_all_users():
    cursor.execute('SELECT id, name, CAST(clear_photo AS TEXT) FROM users')
    users = cursor.fetchall()
    return users

# Route to send all user data
@app.route('/all_users', methods=['GET'])
def send_all_users():
    users = get_all_users()
    return users

# Route to get user logins
@app.route('/getLogins', methods=['GET'])
def get_last_10_logins():
    lock.acquire(True)
    user_id = request.args.get('id')
    cursor.execute('SELECT intime FROM logins WHERE user_id=? ORDER BY intime DESC LIMIT 10', (user_id,))
    res = cursor.fetchall()
    lock.release()
    return res


# Route to merge users
@app.route('/merge', methods=['POST'])
def merge_users_route():
    data = request.json
    primary_user_id = data['primary_user_id']
    secondary_user_ids = data['secondary_user_ids']
    # Call merge_users function here
    merge_users(primary_user_id, secondary_user_ids)
    return "Users merged successfully", 200

if __name__ == '__main__':
    # Start the Flask application
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True,host="192.168.214.8")
    # app.run(debug=True)


    