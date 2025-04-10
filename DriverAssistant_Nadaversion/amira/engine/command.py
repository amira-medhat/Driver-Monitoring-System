# =================== MODULE IMPORTS ===================

# Core system and debugging modules
import os  # For file system access
import sys  # System-specific parameters and functions
import traceback  # To print detailed error stack traces

# Text-to-speech using Google TTS (not actively used, kept as fallback)
from gtts import gTTS  

# Used for playing audio files
import pygame  

# For speech recognition
import speech_recognition as sr

# Whisper model for speech-to-text (currently unused in this file)
import whisper  

# For recording from microphone
import sounddevice as sd  
from scipy.io.wavfile import write  # To save recorded audio
import numpy as np  # For audio data handling

# For frontend-backend interaction
import eel  

# Time-based operations
import time

# LLM interface for local models like LLaMA via Ollama
import ollama

# Multithreading for background tasks
import threading

# JSON file handling
import json

# Asynchronous I/O support
import asyncio

# Offline TTS engine from Microsoft Edge voices
import edge_tts

# For current date/time
from datetime import datetime

# For IP-based location lookup
import geocoder  

# Open URLs in browser
import webbrowser

# Location services and reverse geocoding
import geopy  
from geopy.geocoders import Nominatim

# Audio handling
from pydub import AudioSegment  # For audio decoding
from pydub.playback import play  # To play decoded audio

# Input/Output buffer stream
import io  

# HTTP requests (used by some backend functions)
import requests

# Sending emails
import smtplib  
from send_email import send_alert_email  # Custom module to send alert email

# Configuration variables like WhatsApp contacts
from config import *  


# =================== APP STATE ===================

class AppState:
    """
    Holds the global state of the assistant application.
    """
    def __init__(self):
        self.current_mode = "monitoring"  # Tracks if assistant is monitoring or assisting
        self.mic_pressed = False  # Flag to indicate mic button was pressed
        self.conversation_history = []  # Stores LLM conversation context
        self.location_override = None  # Holds GPS location from frontend if available
        self.json_file_path = "data/driver_alert.json"  # Path to monitoring data
        self.json_flag = True  # If true, JSON monitoring is enabled


# =================== AUDIO MANAGER ===================

class AudioManager:
    """
    Manages audio output, including TTS, buzzer alarms, and music playback.
    """

    def __init__(self):
        self.state = state  # Use the shared app state
        pygame.mixer.init()  # Initialize the Pygame mixer for audio playback
    
    def play(self, path):
        """
        Play an audio file from the given path.
        """
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

    def stop(self):
        """
        Stop any currently playing audio.
        """
        pygame.mixer.music.stop()

    def pause(self):
        """
        Pause the current audio.
        """
        pygame.mixer.music.pause()

    def unpause(self):
        """
        Resume paused audio.
        """
        pygame.mixer.music.unpause()

    def set_volume(self, volume):
        """
        Set the audio volume (0.0 to 1.0).
        """
        pygame.mixer.music.set_volume(volume)
    
    @eel.expose
    def speak(self, text):
        """
        Speak a given text using edge TTS.
        """
        print(f"[TTS] Speaking: {text}")
        asyncio.run(self.edge_speak(text))  # Run the async function synchronously

    async def edge_speak(self, text, voice="en-US-AriaNeural"):
        """
        Convert text to speech using edge-tts and play it with PyDub.
        """
        communicate = edge_tts.Communicate(text=text, voice=voice)
        mp3_bytes = b""  # Buffer for raw audio

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_bytes += chunk["data"]

        # Convert byte stream to playable audio and play
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        play(audio)

    def BuzzerSound(self):
        """
        Play a buzzer alert sound.
        """
        music_path = "www//assets//audio//buzzer.wav"
        self.play(music_path)

        
class LLMManager():
    """
    Handles conversation flow with the user and communicates with the LLM (LLaMA via Ollama).
    """

    def __init__(self, state, Audio, User):
        self.state = state  # Shared app state
        self.User = User  # Handles STT
        self.Audio = Audio  # Handles TTS and audio output

        # Initialize conversation history with system prompt
        self.state.conversation_history = [self.generate_initial_context()]
        
    def generate_initial_context(self):
        """
        Generates the system prompt with time, date, and location for assistant initialization.
        """
        now = datetime.now()
        current_date = now.strftime("%A, %B %d, %Y")
        current_time = now.strftime("%I:%M %p")

        try:
            # Approximate location via IP
            g = geocoder.ip('me')
            city = g.city or "an unknown city"
            country = g.country or "an unknown country"
            location_text = f"You are currently in {city}, {country}."

            # Override location from frontend if available
            if self.state.location_override:
                location_text = f"You are currently at {self.state.location_override}."
        except Exception as e:
            location_text = "Location is unavailable."
            print(f"[ERROR] Failed to get location: {e}")

        # System message that defines the assistant's role
        system_prompt = f"""You are a helpful driving assistant.
Today is {current_date} and the time is {current_time}.
{location_text}
You do functions like providing him current location, current time, traffic and weather info, estimated arrival time to destinations and routing steps to these destinations.
You also talk with him if he feels sleepy or suffers fatigue.
Only respond in 1–2 sentences unless instructed otherwise.
"""

        return {"role": "system", "content": system_prompt}

    def estimate_tokens(self, text):
        """
        Estimates the number of tokens in a string (approx 1.3 tokens per word).
        """
        return len(text.split()) * 1.3

    def trim_history(self, history, max_tokens=7000):
        """
        Keeps conversation within token limits by trimming older messages, retaining the system prompt.
        """
        total_tokens = 0
        trimmed = []

        # Retain the initial system message
        if history and history[0]['role'] == 'system':
            trimmed.append(history[0])
            history = history[1:]

        # Keep the most recent messages up to the token limit
        for msg in reversed(history):
            tokens = self.estimate_tokens(msg['content'])
            if total_tokens + tokens <= max_tokens:
                trimmed.insert(1, msg)  # Insert after system message
                total_tokens += tokens
            else:
                break  # Stop adding if we hit the token limit

        return trimmed

    @eel.expose
    def PassToLlm(self):
        """
        Called when assistant is in 'assistance' mode to handle user queries and reply via LLM.
        """
        silence_timeout = 30  # Timeout to return to monitoring mode if user is silent
        last_response_time = time.time()

        while self.state.current_mode == "assistance":
            query = self.User.takecommand()  # Get speech input
            print(f"[USER]: {query}")

            # If no speech detected
            if query == "none" :
                if time.time() - last_response_time > silence_timeout:
                    eel.DisplayMessage("No response detected. Going back to monitoring mode.")
                    self.Audio.speak("No response detected. Going back to monitoring mode.")
                    self.state.current_mode = "monitoring"
                    self.state.mic_pressed = False
                    eel.ExitHood()
                    break
                continue  # Keep trying if still within timeout

            # Reset silence timer
            last_response_time = time.time()

            # Add user message to conversation
            user_message = {"role": "user", "content": query}
            self.state.conversation_history.append(user_message)



            #----------------------------Check for close gps phrases---------------------------------------
            if any(phrase in query for phrase in ["close gps", "stop gps", "turn off gps", "hide","hide gps","close map","stop map"]):
                print("[DEBUG] closing google maps website returning to assistant home screen.")
                self.Audio.speak("Got it !")
                from engine.features import CloseMaps
                CloseMaps()
                self.state.current_mode = "monitoring"
                self.state.mic_pressed = False
                eel.ExitHood()
                break

            # Check for open gps phrases
            if any(phrase in query for phrase in ["open gps", "turn on gps", "open maps","open"]):
                print("[DEBUG] opening google maps website.")
                from engine.features import OpenGps
                OpenGps("gps")
                self.state.current_mode = "monitoring"
                self.state.mic_pressed = False
                eel.ExitHood()
                break

            # ======================== Exit Phrases ============================
            if any(phrase in query for phrase in ["goodbye", "bye", "thank you", "thanks", "exit", "end", "close"]):
                print("[DEBUG] Switching to monitoring mode and ending chatting.")
                eel.DisplayMessage("Goodbye driver !")
                self.Audio.speak("Goodbye driver !")
                self.state.current_mode = "monitoring"
                self.state.mic_pressed = False
                eel.ExitHood()
                break

            # =================== Back to Monitoring ==========================
            if any(phrase in query for phrase in ["enable monitoring", "monitoring mode", "back to monitoring", "start monitoring", "start monitor", "enable"]):
                print("[DEBUG] Switching again to monitoring mode.")
                eel.DisplayMessage("Got it!")
                self.Audio.speak("Got it!")
                eel.DisplayMessage("Switching to monitoring mode.")
                self.Audio.speak("Switching to monitoring mode.")
                self.state.current_mode = "monitoring"
                self.state.mic_pressed = False
                self.state.json_flag = True
                eel.ExitHood()
                break

            # =================== Disable Monitoring ==========================
            if any(phrase in query for phrase in ["send", "text", "message", "whatsapp", "call", "voice call", "make a call", "ring"]):
                self.state.current_mode = "assistance"
                is_call = any(word in query for word in ["call", "voice call", "make a call", "ring"])
                action_type = "call" if is_call else "message"

                # Contact dictionary
                contacts = {
                    "nada": "+201093661321",
                    "mama": "+201270509918",
                    # Add more contacts here
                }

                # Attempt to get name from query
                name = None
                for contact_name in contacts:
                    if contact_name in query:
                        name = contact_name
                        break

                # If name not found, ask once more
                if name is None:
                    self.Audio.speak("sorry i didnt catch the name can you repeat it")
                    name_attempt = self.User.takecommand()
                    for contact_name in contacts:
                        if contact_name in name_attempt:
                            name = contact_name
                            break

                    if name is None:
                        self.Audio.speak("I still can't catch the name...leaving")
                        return  # Exit this section

                number = contacts[name]
                from engine.features import send_whatsApp_msg

                # === CALL HANDLING ===
                if action_type == "call":
                    self.Audio.speak(f"Calling {name} on WhatsApp.")
                    send_whatsApp_msg(number, message="", flag="call", name=name)
                    continue

                # === MESSAGE HANDLING ===
                else:
                    self.Audio.speak(f"Got it. Now tell me the message to send to {name}.")
                    eel.DisplayMessage(f"Got it. Now tell me the message to send to {name}.")

                    message = "none"
                    start_time = time.time()
                    timeout = 25  # seconds
                    retry_prompt_given = False

                    while message == "none" and (time.time() - start_time) < timeout:
                        message = self.User.takecommand()

                        if message == "none" and not retry_prompt_given:
                            self.Audio.speak("Please say it again.")
                            retry_prompt_given = True

                    if message == "none":
                        self.Audio.speak("No message received...leaving .")
                        return  # Exit this section

                    send_whatsApp_msg(number, message, flag="message", name=name)
                    self.Audio.speak("Message sent.")
                    continue

            # =================== Navigation Request ============================
            if query.lower().startswith("navigate to"):
                self.handle_navigation(query)
                continue

            # =================== Send Other Requests to LLM ====================
            try:
                safe_history = self.trim_history(self.state.conversation_history)
                response = ollama.chat(model='llama3.2', messages=safe_history)
                llama_response = response['message']['content']
                eel.DisplayMessage(llama_response)
                self.Audio.speak(llama_response)
                assistant_message = {"role": "assistant", "content": llama_response}
                self.state.conversation_history.append(assistant_message)

            except Exception as e:
                print(f"[ERROR] LLM call failed: {e}")
                eel.DisplayMessage("Sorry, I couldn't process your request.")
                self.Audio.speak("Sorry, I couldn't process your request.")
                time.sleep(1)

    def handle_navigation(self, query):
        """
        Opens Google Maps with the specified destination.
        """
        destination = query.lower().replace("navigate to", "").strip()
        if destination:
            eel.DisplayMessage(f"Opening directions to {destination.title()}...")
            self.Audio.speak(f"Opening directions to {destination.title()}...")
            webbrowser.open(f"https://www.google.com/maps/dir/?api=1&destination={destination.replace(' ', '+')}")
        else:
            self.Audio.speak("Where would you like to go?")

        
class UserManager:
    def __init__(self, state, Audio):
        self.state = state  # Reference to global app state
        self.Audio = Audio  # Reference to AudioManager for playing voice responses

    @eel.expose
    def takecommand(self, timeout=15, phrase_time_limit=20):
        """
        Listens to the user's voice and converts it to text.
        Returns recognized text in lowercase, or 'none' if failed.
        """
        recognizer = sr.Recognizer()  # Speech recognizer instance
        mic = sr.Microphone()  # Use system microphone

        with mic as source:
            print("[STT] Listening for a command...")
            eel.DisplayMessage("[Listening for command...]")

            recognizer.adjust_for_ambient_noise(source)  # Reduce background noise

            try:
                # Listen for input with timeout limits
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                print("[STT] Recognizing...")
                eel.DisplayMessage("[Recognizing...]")

                # Recognize speech using Google API
                query = recognizer.recognize_google(audio, language="en")
                print(f"[STT] You said: {query}")
                return query.lower()

            except sr.WaitTimeoutError:
                # Timeout without speech
                print("[STT] No speech detected (timeout)")
                eel.DisplayMessage("[STT Timeout — no speech detected]")
                return "none"

            except sr.UnknownValueError:
                # Speech unclear
                print("[STT] Could not understand the audio")
                eel.DisplayMessage("[Could not understand the audio]")
                return "none"

            except sr.RequestError:
                # API issue (likely no internet)
                print("[STT] STT request failed (possibly no internet)")
                eel.DisplayMessage("[STT request failed (check internet)]")
                return "none"

    @eel.expose
    def ListenForWakeWord(self, wake_word="hello no"):
        """
        Listens for a specific wake word. If detected, returns True.
        Otherwise, returns False.
        """
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("[Listening for wake word...]")
            recognizer.adjust_for_ambient_noise(source)

            try:
                # Listen for wake word
                audio = recognizer.listen(source, timeout=15, phrase_time_limit=7)
                transcript = recognizer.recognize_google(audio).lower()
                print(f"[DEBUG] Heard: {transcript}")

                if wake_word in transcript:
                    # Wake word matched
                    eel.DisplayMessage("Hey driver, how can I help you?")
                    self.Audio.speak("Hey driver, how can I help you?")
                    time.sleep(0.5)
                    return True

            except sr.WaitTimeoutError:
                print("[DEBUG] No speech detected in time.")
            except sr.UnknownValueError:
                print("[DEBUG] Could not understand the audio.")
            except sr.RequestError as e:
                print(f"[ERROR] Speech recognition failed: {e}")

        return False  # Wake word not detected

    def send_feedback_to_EC(self):
        """
        Notifies emergency contact by email and WhatsApp if driver is unresponsive.
        """
        send_alert_email(LIVE_STREAM_URL, RECEIVER_EMAIL)  # Send email with livestream link

        from engine.features import send_whatsApp_msg
        # Send WhatsApp message
        send_whatsApp_msg(
            EMERGENCY_CONTACT_NUMBER,
            "🚨 *Emergency Alert!* 🚨\nDriver seems to be unresponsive 😔\n📺 Check the live stream here:\n🔗 " + LIVE_STREAM_URL,
            'message',
            EMERGENCY_CONTACT_NAME
        )

        time.sleep(1)  # Wait briefly

        # Trigger WhatsApp call
        send_whatsApp_msg(
            EMERGENCY_CONTACT_NUMBER,
            "",
            'call',
            EMERGENCY_CONTACT_NAME
        )

    def alert(self):
        """
        Triggered by fatigue detection. Asks driver if they're okay.
        If no response, sends emergency alert.
        """
        self.state.current_mode = "fatigue_alert"
        eel.DisplayMessage("Are you okay? Can you hear me?")
        self.Audio.speak("Are you okay? Can you hear me?")
        time.sleep(0.5)
        eel.ShowHood()

        # Wait for response
        response = self.check_up(timeout=50)

        if response is None:
            # No response — alert emergency contact
            eel.DisplayMessage("Dangerous! No response from driver.")
            self.Audio.speak("Dangerous! No response from driver.")
            time.sleep(0.5)
            self.Audio.BuzzerSound()
            time.sleep(0.5)
            self.send_feedback_to_EC()
            time.sleep(7)
            self.state.current_mode = "monitoring"
            eel.ExitHood()
            return

        else:
            # Driver responded — ask if they want to talk
            eel.DisplayMessage("Do you want to talk to me?")
            self.Audio.speak("Do you want to talk to me?")
            time.sleep(0.5)
            answer = self.check_up(timeout=40)

            if answer and "yes" in answer:
                eel.DisplayMessage("Okay, I'm here to help you , you're in assistance mode now")
                self.Audio.speak("Okay, I'm here to help you , you're in assistance mode now")
                time.sleep(0.5)
                eel.ShowHood()
                self.state.current_mode = "assistance"
            else:
                eel.DisplayMessage("okay, let me know if you need me")
                self.Audio.speak("okay, let me know if you need me")
                time.sleep(0.5)
                self.state.current_mode = "monitoring"
                eel.ExitHood()

    def check_up(self, timeout=40):
        """
        Listen for any response from the user within the timeout.
        """
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            eel.DisplayMessage("[Listening for response...]")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                response = recognizer.recognize_google(audio).lower()
                print(f"[Heard]: {response}")
                return response
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                return None

            
# Create instances of main classes
state = AppState()
Audio = AudioManager()
User = UserManager(state, Audio)
LLM = LLMManager(state, Audio, User)

@eel.expose
def ReceiveLocation(lat, lon):
    """
    Receives latitude and longitude from frontend and reverse geocodes it to a readable address.
    """
    try:
        geolocator = Nominatim(user_agent="driver_assistant")
        location = geolocator.reverse((lat, lon), language="en")
        address = location.address
        # after getting current address put it in json file to use it later
        # Save lat, lon, and address to a file
        with open("location.json", "w") as f:
            json.dump({
                "latitude": lat,
                "longitude": lon,
                "address": address
            }, f)
        print(f"[📍] Precise location: {address}")
        
        state.location_override = address

        # Update the system prompt with new location
        if state.conversation_history and state.conversation_history[0]["role"] == "system":
            state.conversation_history[0] = LLM.generate_initial_context()
        else:
            print("[WARN] Couldn't update system message — history not ready.")
        
    except Exception as e:
        print(f"[ERROR] Failed to reverse geocode: {e}")

@eel.expose
def set_mic_pressed():
    """
    Marks mic as pressed from frontend event.
    """
    state.mic_pressed = True
@eel.expose
def Set_jason_flag():
    state.json_flag = True
    Audio.speak("Monitoring is enabled.")
    
@eel.expose
def Clear_jason_flag():
    state.json_flag = False
    Audio.speak("Monitoring is disabled.")

@eel.expose
def get_monitor_mode():
    return "on" if state.json_flag else "off"

@eel.expose
def monitoring_loop():
    """
    Background loop that runs in the monitoring mode:
    - Checks driver alert JSON for fatigue or distraction
    - Detects wake word or mic press
    - Sends data to LLM for safety suggestions if necessary
    """
    while True:
        # Load the driver alert JSON if it exists
        if os.path.exists(state.json_file_path):
            with open(state.json_file_path, "r") as f:
                data = json.load(f)

        # If driver is sleepy or fatigued → trigger alert flow
        if data.get("Sleep Alert", "").lower() == "true" or data.get("Fatigue Alert", "").lower() == "on":
            state.current_mode == "Fatigue_Alert"  # (Note: should be = not ==)
            User.alert()  # Trigger alert dialog
            time.sleep(5)  # Pause to avoid spamming
            continue

        # If mic was pressed or wake word heard while monitoring → switch to assistance mode
        if state.mic_pressed or (state.current_mode == "monitoring" and User.ListenForWakeWord()):
            state.current_mode = "assistance"
            eel.ShowHood()  # Show assistant UI
            LLM.PassToLlm()  # Start listening to driver
            continue

        # If still in assistance mode (after wake word), continue handling conversation
        if state.current_mode == "assistance":
            eel.ShowHood()
            LLM.PassToLlm()
            continue

        # Normal passive monitoring logic when user hasn’t triggered assistant
        if state.current_mode == "monitoring" and not state.mic_pressed and state.json_flag:
            print(f"[DEBUG] json flag: {state.json_flag}")

            # Re-read the JSON to get the most updated status
            if os.path.exists(state.json_file_path):
                with open(state.json_file_path, "r") as f:
                    data = json.load(f)

                    # If user mode changed mid-read, stop
                    if state.current_mode != "monitoring":
                        continue

                    # Parse the alerts into lowercase strings
                    action = data.get("Activity Alert", "").lower()
                    hands = data.get("HOW Alert", "").lower()
                    health = data.get("Health Alert", "").lower()
                    distraction = data.get("Distraction Alert", "").lower()

                    # Still double-check mode and mic again before proceeding
                    if state.current_mode == "monitoring" and not state.mic_pressed:
                        if state.current_mode != "monitoring":
                            continue

                        # If all is good — safe driving detected — do nothing
                        if action == "safe driving" and distraction == "off" and hands == "on_wheel" and health == "off":
                            continue

                        # Otherwise, send a summarized prompt to LLM for polite safety tip
                        json_alert = json.dumps(data, indent=2)  # Pretty format JSON
                        prompt = (
                            f"Driver activity: {action}. "
                            f"Distraction alert: {distraction}. "
                            f"Hands on or off wheel: {hands}. "
                            f"Health alert: {health}. "
                            f"Based on these observations, provide a short, polite safety instruction in 20 words or less. "
                            f"The tone should be clear and supportive. Avoid generic advice."
                        )

                        try:
                            # Wrap it as a user message for LLM
                            safe_history = [{"role": "user", "content": prompt}]
                            response = ollama.chat(model='llama3.2', messages=safe_history)
                            reply = response['message']['content']

                            print(f"[LLM] Assistant response: {reply}")
                            eel.DisplayMessage(reply)  # Show on frontend
                            Audio.speak(reply)         # Say it aloud

                        except Exception as e:
                            # Catch LLM issues
                            print(f"[ERROR] LLM failed during monitoring: {e}")
                            eel.DisplayMessage("Error analyzing driver alert data.")
                            Audio.speak("Error analyzing driver alert data.")

        # Wait a short time before next loop
        time.sleep(0.1)
