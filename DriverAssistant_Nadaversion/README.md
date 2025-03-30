# 🚘 Wiki — Voice-Based Driver Assistant

**Wiki** is a voice-enabled driver assistant designed to run offline with local LLM (LLaMA 3.2 via Ollama), supporting wake word activation, safety alerts, and a custom desktop dashboard interface. It intelligently interacts with the driver and monitors for fatigue or distraction using external input (JSON alerts).

---

## 🧠 Features

- 🔊 Wake word detection: “Hey Wiki”
- 🎙️ Manual mic activation via GUI button
- 📄 Fatigue & sleep alert integration via JSON file
- 🧠 Local LLM (LLaMA 3.2 via Ollama) to answer queries
- 💬 Text-to-speech response (gTTS)
- 🌐 Web-based dashboard (HTML + Bootstrap + jQuery)
- 🪄 Animated UI using Particles.js and Textillate.js

---

## 📦 Setup

1. **Clone the repo** and navigate into the project folder.

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv llm_env
   source llm_env/bin/activate  # For Linux/macOS
   # OR
   llm_env\Scripts\activate  # For Windows

3. **Run code**:
```bash
python main.py

