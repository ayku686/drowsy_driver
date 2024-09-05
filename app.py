from flask import Flask, render_template, jsonify
import subprocess
import sys
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    venv_python = os.path.join(sys.prefix, 'bin', 'python') if os.name != 'nt' else os.path.join(sys.prefix, 'Scripts', 'python.exe')
    script_path = os.path.join(os.path.dirname(__file__), 'detect.py')
    subprocess.Popen([venv_python, script_path])
    return jsonify({'message': 'Drowsiness detection started!'})

if __name__ == '__main__':
    app.run(debug=True)
