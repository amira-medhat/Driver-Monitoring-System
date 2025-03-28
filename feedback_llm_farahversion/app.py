from flask import Flask, jsonify, render_template, request
import json
import pyttsx3
import requests

app = Flask(__name__)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

@app.route("/analyze", methods=["GET"])
def analyze():
    with open("driver_state.json", "r") as f:
        driver_data = json.load(f)

    if driver_data.get("Sleep Alert") == "true":
        speak("Hey, are you there?")
        return jsonify({"mode": "sleep", "message": "dangerous"})

    prompt = (
        f"Driver activity: {driver_data.get('Activity Alert', 'none')}, "
        f"Distraction: {driver_data.get('Distraction alert', 'none')}, "
        f"Fatigue: {driver_data.get('Fatigue Alert', 'none')}. "
        "Generate a 20-word safety instruction based on the above data. Be concise."
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": prompt, "stream": False}
        )
        result = response.json()
        safety_instruction = ' '.join(result.get("response", "").strip().split()[:20])
        return jsonify({"mode": "safety", "message": safety_instruction})  # No speak here!
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/neon_assistant", methods=["POST"])
def neon_assistant():
    data = request.get_json()
    user_input = data.get("question", "")
    if not user_input:
        return jsonify({"response": "Sorry, I didn't hear anything."})

    prompt = f"Answer briefly and clearly: {user_input}"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": prompt, "stream": False}
        )
        result = response.json()
        reply = result.get("response", "").strip()
        # speak(reply)
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
