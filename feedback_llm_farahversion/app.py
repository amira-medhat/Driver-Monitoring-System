from flask import Flask, jsonify, render_template, request
import json
import pyttsx3
import requests
import smtplib
from email.mime.text import MIMEText


app = Flask(__name__)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()



@app.route("/send_email", methods=["POST"])
def send_email():
    data = request.get_json()
    youtube_link = data.get("link", "")

    sender_email = "farahhahmed01@gmail.com"
    receiver_email = "omarrida5@gmail.com"
    app_password = ""  # Your Gmail App Password
    subject = "🚨 Driver Unresponsive Alert"
    body = f"""
    <html>
    <body style="margin: 0; padding: 0; background-color: #f0f2f5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <div style="max-width: 600px; margin: 30px auto; background: #ffffff; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); overflow: hidden;">

        <!-- Header -->
        <div style="background-color: #dc3545; padding: 20px;">
            <h2 style="color: #ffffff; margin: 0;">🚨 Driver Safety Alert</h2>
        </div>

        <!-- Body -->
        <div style="padding: 25px;">
            <p style="font-size: 16px; color: #333;">
            <strong>Alert:</strong> The driver appears to be <span style="color: red;"><strong>unresponsive</strong></span>. Immediate action is recommended.
            </p>
            <p style="font-size: 15px; color: #555;">
            We suggest trying to call the driver directly to confirm their condition. If you are unable to reach them, please review the live video feed from inside the car.
            </p>

            <!-- Button -->
            <div style="text-align: center; margin: 30px 0;">
            <a href="{youtube_link}" target="_blank" 
                style="background-color: #007bff; color: white; padding: 12px 24px; text-decoration: none; font-weight: bold; border-radius: 5px; display: inline-block;">
                ▶️ View Live Stream
            </a>
            </div>

            <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">

            <p style="font-size: 13px; color: #888; text-align: center;">
            Sent automatically by your Driver Monitoring System.<br>
            Please do not reply to this email.
            </p>
        </div>
        </div>
    </body>
    </html>
    """

    msg = MIMEText(body, "html")  # ✅ Tells email client to render as HTML
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        return jsonify({"status": "Email sent successfully"})
    except Exception as e:
        return jsonify({"error": str(e)})

    

@app.route("/analyze", methods=["GET"])
def analyze():
    try:
        with open("driver_state.json", "r") as f:
            driver_data = json.load(f)
    except Exception as e:
        return jsonify({"mode": "error", "message": f"Could not load driver_state.json: {str(e)}"})

    # Check for sleep alert
    if driver_data.get("Sleep Alert") == "true":
        speak("Hey, are you there?")
        return jsonify({"mode": "sleep", "message": "dangerous"})

    # Construct LLM prompt
    activity = driver_data.get("Activity Alert", "none")
    distraction = driver_data.get("Distraction alert", "none")
    fatigue = driver_data.get("Fatigue Alert", "none")

    prompt = (
        f"Driver activity: {activity}. "
        f"Distraction level: {distraction}. "
        f"Fatigue status: {fatigue}. "
        f"Generate a concise 20-word maximum safety instruction based on these observations."
    )

    print("[DEBUG] Prompt sent to LLM:\n", prompt)

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": prompt, "stream": False}
        )
        result = response.json()
        full_response = result.get("response", "").strip()
        print("[DEBUG] Raw LLM response:", full_response)

        safety_instruction = ' '.join(full_response.split()[:20])
        if not safety_instruction:
            safety_instruction = "empty."

        return jsonify({"mode": "safety", "message": safety_instruction})

    except Exception as e:
        print("[ERROR] LLM request failed:", str(e))
        return jsonify({"mode": "error", "message": f"LLM request failed: {str(e)}"})

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
