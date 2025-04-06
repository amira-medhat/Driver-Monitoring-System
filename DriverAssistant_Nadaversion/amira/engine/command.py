import os
import sys
import traceback
from gtts import gTTS  # Google Text-to-Speech
import pygame
import speech_recognition as sr
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import eel
import time
import ollama
import threading
import json
import asyncio
import edge_tts
from datetime import datetime
import geocoder
import webbrowser
import geopy
from geopy.geocoders import Nominatim
from pydub import AudioSegment
from pydub.playback import play
import io

class AppState:
    def __init__(self):
        self.current_mode = "monitoring"
        self.mic_pressed = False
        self.conversation_history = []
        self.location_override = None
        self.json_file_path = "data/driver_alert.json"



class AudioManager:
    
    def __init__(self, state):
        self.state = state
        pygame.mixer.init()
    
    def play(self, path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

    def stop(self):
        pygame.mixer.music.stop()

    def pause(self):
        pygame.mixer.music.pause()

    def unpause(self):
        pygame.mixer.music.unpause()

    def set_volume(self, volume):
        pygame.mixer.music.set_volume(volume)
    
    @eel.expose
    def speak(self, text):
        print(f"[TTS] Speaking: {text}")
        asyncio.run(self.edge_speak(text))

            
    async def edge_speak(self, text, voice="en-US-AriaNeural"):
        communicate = edge_tts.Communicate(text=text, voice=voice)
        mp3_bytes = b""

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_bytes += chunk["data"]

        # Convert to playable audio
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        play(audio)


    def BuzzerSound(self):
        music_path = "www//assets//audio//buzzer.wav"
        self.play(music_path)
        
class LLMManager():
    def __init__(self, state, Audio, User):
        self.state = state
        self.User = User
        self.Audio = Audio
        self.state.conversation_history = [self.generate_initial_context()]
        
    def generate_initial_context(self):
        now = datetime.now()
        current_date = now.strftime("%A, %B %d, %Y")
        current_time = now.strftime("%I:%M %p")

        # Get approximate location (based on IP)
        try:
            g = geocoder.ip('me')
            city = g.city or "an unknown city"
            country = g.country or "an unknown country"
            location_text = f"You are currently in {city}, {country}."
            if self.state.location_override:
                location_text = f"You are currently at {self.state.location_override}."
        except Exception as e:
            location_text = "Location is unavailable."
            print(f"[ERROR] Failed to get location: {e}")

        system_prompt = f"""You are a helpful driving assistant.
Today is {current_date} and the time is {current_time}.
{location_text}
You do functions like providing him current location, current time, traffic and weather info, estimated arrival time to destinations and routing steps to these destinations.
You also talk with him if he feels sleepy or suffers fatigue.
Only respond in 1–2 sentences unless instructed otherwise.
"""

        return {"role": "system", "content": system_prompt}

            
    def estimate_tokens(self, text):
        return len(text.split()) * 1.3  # ≈ 1.3 tokens per word (safe estimate)

    def trim_history(self, history, max_tokens=7000):  # Leave room for response
        total_tokens = 0
        trimmed = []

        # Keep system message at the top
        if history and history[0]['role'] == 'system':
            trimmed.append(history[0])
            history = history[1:]

        # Add from most recent to oldest, until we reach token limit
        for msg in reversed(history):
            tokens = self.estimate_tokens(msg['content'])
            if total_tokens + tokens <= max_tokens:
                trimmed.insert(1, msg)  # insert after system message
                total_tokens += tokens
            else:
                break  # stop when too many

        return trimmed
    
    @eel.expose
    def PassToLlm(self):

        # Continue processing commands until the user issues an exit command
        while self.state.current_mode == "assistance":
            query = self.User.takecommand()  # Listen for a command
            print(f"[USER]: {query}")
            user_message = {"role": "user", "content": query}
            self.state.conversation_history.append(user_message)

            # If nothing was heard, notify the user and continue the loop
            if query == "none":
                eel.DisplayMessage("I didn't hear anything. Can you repeat that?")
                self.Audio.speak("I didn't hear anything. Can you repeat that?")
                time.sleep(0.5)
                continue  # Retry listening for a valid command

            # Check if the user wants to exit assistance mode
            if any(phrase in query for phrase in ["enable monitoring", "monitoring mode", "back to monitoring", "exit", "disable assistant", "monitoring", "disable assist"]):
                print("[DEBUG] Switching to monitoring mode.")
                eel.DisplayMessage("Got it!")
                self.Audio.speak("Got it!")
                eel.DisplayMessage("Switching to monitoring mode.")
                self.Audio.speak("Switching to monitoring mode.")
                self.state.current_mode = "monitoring"
                self.state.mic_pressed = False
                eel.ExitHood()
                break  # Exit the loop
            
            # Check for navigation command
            if query.lower().startswith("navigate to"):
                self.handle_navigation(query)
                continue  # skip sending to LLM, since we handled it

            # Otherwise, process the command via LLM
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

        # End of assistance mode: now in monitoring mode.
        # (If the loop exits normally, we assume an exit command was given.)
        # Any necessary cleanup can be done here.
        
    def handle_navigation(self, query):
        # Extract destination
        destination = query.lower().replace("navigate to", "").strip()
        if destination:
            eel.DisplayMessage(f"Opening directions to {destination.title()}...")
            self.Audio.speak(f"Opening directions to {destination.title()}...")

            # Open in default browser
            webbrowser.open(f"https://www.google.com/maps/dir/?api=1&destination={destination.replace(' ', '+')}")
        else:
            self.Audio.speak("Where would you like to go?")
        
class UserManager:
    def __init__(self, state, Audio):
        self.state = state
        self.Audio = Audio
    
    @eel.expose
    def takecommand(self, timeout=15, phrase_time_limit=20):
        """Listens for speech with timeout and returns recognized text or 'none'."""
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            print("[STT] Listening for a command...")
            eel.DisplayMessage("[Listening for command...]")

            # Adjust to ambient noise
            recognizer.adjust_for_ambient_noise(source)

            try:
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                print("[STT] Recognizing...")
                eel.DisplayMessage("[Recognizing...]")

                query = recognizer.recognize_google(audio, language="en")
                print(f"[STT] You said: {query}")
                return query.lower()

            except sr.WaitTimeoutError:
                print("[STT] No speech detected (timeout)")
                eel.DisplayMessage("[STT Timeout — no speech detected]")
                return "none"

            except sr.UnknownValueError:
                print("[STT] Could not understand the audio")
                eel.DisplayMessage("[Could not understand the audio]")
                return "none"

            except sr.RequestError:
                print("[STT] STT request failed (possibly no internet)")
                eel.DisplayMessage("[STT request failed (check internet)]")
                return "none"

    @eel.expose
    def ListenForWakeWord(self, wake_word="hey man"): 

        # global current_mode
        recognizer = sr.Recognizer()


        with sr.Microphone() as source:
            print("[Listening for wake word...]")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=15, phrase_time_limit=7)
                transcript = recognizer.recognize_google(audio).lower()
                print(f"[DEBUG] Heard: {transcript}")

                if wake_word in transcript:
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

        return False
    
    def alert(self):
        self.state.current_mode = "fatigue_alert"
        eel.DisplayMessage("Are you okay? Can you hear me?")
        self.Audio.speak("Are you okay? Can you hear me?")
        time.sleep(0.5)
        eel.ShowHood()
        response = self.check_up(timeout=10)

        if response is None:
            eel.DisplayMessage("Dangerous! No response from driver.")
            self.Audio.speak("Dangerous! No response from driver.")
            time.sleep(0.5)
            self.Audio.BuzzerSound()
            self.state.current_mode = "monitoring"
            eel.ExitHood()
            return
        else:
            eel.DisplayMessage("Do you want to talk to me?")
            self.Audio.speak("Do you want to talk to me?")
            time.sleep(0.5)
            answer = self.check_up(timeout=10)
            if answer and "yes" in answer:
                eel.DisplayMessage("Okay, I'm here to help you.")
                self.Audio.speak("Okay, I'm here to help you.")
                time.sleep(0.25)
                eel.DisplayMessage("you're in assistance mode now")
                self.Audio.speak("you're in assistance mode now")
                time.sleep(0.5)
                eel.ShowHood()
                self.state.current_mode = "assistance"
            else:
                eel.DisplayMessage("Okay, I'm here if you need anything. Goodbye!")
                self.Audio.speak("Okay, I'm here if you need anything. Goodbye!")
                time.sleep(0.5)
                self.state.current_mode = "monitoring"
                eel.ExitHood()

    def check_up(self, timeout=10):
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
            
            
state = AppState()   
Audio = AudioManager(state)
User = UserManager(state, Audio)
LLM = LLMManager(state, Audio, User)

@eel.expose
def ReceiveLocation(lat, lon):
    try:
        geolocator = Nominatim(user_agent="driver_assistant")
        location = geolocator.reverse((lat, lon), language="en")
        address = location.address
        print(f"[📍] Precise location: {address}")
        
        state.location_override = address
        # Update initial assistant prompt with accurate location
        # Only replace if history exists
        if state.conversation_history and state.conversation_history[0]["role"] == "system":
            state.conversation_history[0] = LLM.generate_initial_context()
        else:
            print("[WARN] Couldn't update system message — history not ready.")
        
    except Exception as e:
        print(f"[ERROR] Failed to reverse geocode: {e}")

@eel.expose
def set_mic_pressed():
    state.mic_pressed = True

@eel.expose
def monitoring_loop():
    while True:
        # Enter assistant mode via wake word or mic
        if state.mic_pressed or (state.current_mode == "monitoring" and User.ListenForWakeWord()):
            state.current_mode = "assistance"
            eel.ShowHood()
            LLM.PassToLlm()
            continue
        
        if state.current_mode == "assistance":
            eel.ShowHood()
            LLM.PassToLlm()
            continue

        # Run only if in monitoring mode
        if state.current_mode == "monitoring" and not state.mic_pressed:
            if os.path.exists(state.json_file_path):
                with open(state.json_file_path, "r") as f:
                    data = json.load(f)

                    # Mode may change mid-read, check again
                    if state.current_mode != "monitoring":
                        continue

                    fatigue = data.get("Fatigue Alert", "").lower()
                    sleep = data.get("Sleep Alert", False)
                    action = data.get("Activity Alert", "").lower()
                    hands = data.get("HOW Alert", "").lower()
                    health = data.get("Health Alert", "").lower()
                    distraction = data.get("Distraction Alert", "").lower()

                    # Speak alerts only if still in monitoring mode
                    if state.current_mode != "monitoring":
                        continue

                    if fatigue == "on" or sleep:
                        Audio.speak("[ALERT] Fatigue or Sleep Alert detected.")
                        User.alert()
                    if action != "safe driving" and distraction == "on":
                        Audio.speak("please drive safely and focus on the road")
                    if hands == "off_wheel":
                        Audio.speak("Please keep your hands on the wheel")
                    if health == "on":
                        Audio.speak("Please take care as your heart rate is high. Take a break if needed")
                    if action == "eating":
                        Audio.speak("Please avoid eating while driving")
                    if action == "drinking":
                        Audio.speak("Please avoid drinking while driving")
                    if action in ["talking on the phone", "texting on the phone"]:
                        Audio.speak("Please avoid using phone while driving")

        time.sleep(0.1)