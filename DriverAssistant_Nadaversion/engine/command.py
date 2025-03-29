import os
from gtts import gTTS  # Google Text-to-Speech
import pygame
import speech_recognition as sr
import eel
import time
import ollama
import threading
import json

current_mode = "monitoring" # initial mode
mic_pressed = False
json_file_path = "data/driver_alert.json"  # path where Jetson Nano writes JSON


################## another way for text to speech #####################
# import pyttsx3
# def speak(text):
#     # Generate speech with gTTS
#     engine = pyttsx3.init()
#     #voices = engine.getProperty('voices')
#     #engine.setProperty('voice', voices[0].id)
#     #print(voices)
#     engine.setProperty('rate', 125)
#     engine.say(text)
#     engine.runAndWait()

# speak("hello nada how are you today ?") 

@eel.expose
def speak(text):
    """ Convert Arabic text to speech using gTTS """
    #global stop_speaking
    stop_speaking = False
    print(f"Speaking in English: {text}")

    # Generate speech with gTTS
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    # Play the generated speech
    #os.system("start response.mp3")  # Windows

    #for linux/windows
    music_path = "response.mp3"
    pygame.mixer.init()
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

@eel.expose
def takecommand():

    r=sr.Recognizer()
    with sr.Microphone() as source:
        eel.DisplayMessage("Listening....")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)

        #Wait up to 10 seconds for me to start talking,and once I start, record only 6 seconds of what I say
        audio=r.listen(source,10,6)
    try:
        eel.DisplayMessage("Recognizing....")
        query=r.recognize_google(audio,language='en')
     

    except Exception as e:
        eel.DisplayMessage("Could not hear you :( ,Say that again please....")
        time.sleep(3)
        query="None"
        #takecommand()

    return query.lower()



@eel.expose
def PassToLlm():

    global mic_pressed
    global current_mode
    
    text = takecommand()
    eel.DisplayMessage(text)
    #voice command to stop the monitoring
    if any(phrase in text.lower() for phrase in ["disable moinitoring","disable monitor","disable mon", "stop monitoring","stop monitor","stop mon"]):
        speak("Monitoring disabled.")
        current_mode = "no_monitoring"
        mic_pressed = False
        eel.ExitHood()
        return   # Exit the function (stop recursion)

    #voice command to start the monitoring
    if any(phrase in text.lower() for phrase in ["enable monitoring","enable monitor","enable mon","start mon","start monitor", "start monitoring"]):
        speak("Monitoring enabled.")
        current_mode = "monitoring"
        mic_pressed = False
        eel.ExitHood()
        return   # Exit the function (stop recursion)
        
    # Exit condition
    if any(phrase in text.lower() for phrase in ["bye", "thank you", "thanks", "goodbye",
    "exit","close","shut down","shut up","shut it down"]):
        speak("Goodbye!")
        eel.ExitHood()
        mic_pressed = False
        return  # Exit the function (stop recursion)
    

    response = ollama.chat(model='llama3.2', messages=[{
    'role': 'user',
    'content': f"{text}\n\nPlease answer in 15 words or fewer."}])

    llama_response = response['message']['content']
    
    eel.DisplayMessage(" ")
    speak(llama_response)
    eel.ShowHood()
    time.sleep(1)
    PassToLlm()
    
@eel.expose
def set_mic_pressed():
    global mic_pressed
    mic_pressed = True

    
@eel.expose
def ListenForWakeWord(wake_word="hey wiki"): 

    recognizer = sr.Recognizer()
    global current_mode

    with sr.Microphone() as source:
        print("[Listening for wake word...]")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
            transcript = recognizer.recognize_google(audio).lower()
            print(f"[DEBUG] Heard: {transcript}")

            if wake_word in transcript:
                speak("Hey driver, how can I help you?")
                return True

        except sr.WaitTimeoutError:
            print("[DEBUG] No speech detected in time.")
        except sr.UnknownValueError:
            print("[DEBUG] Could not understand the audio.")
        except sr.RequestError as e:
            print(f"[ERROR] Speech recognition failed: {e}")

    return False 

@eel.expose
def monitoring_loop():

    global current_mode
    global mic_pressed
    
    while True:
        if mic_pressed:
            while mic_pressed:
                PassToLlm()

            continue

        if ListenForWakeWord():
            break
        
        #Check if new JSON alert is written
        if os.path.exists(json_file_path) and current_mode == "monitoring" and mic_pressed == False:
            with open(json_file_path, "r") as f:
                data = json.load(f)

                #get data from JSON file
                fatigue = data.get("Fatigue Alert", "").lower()
                sleep = data.get("Sleep Alert", False)
                action = data.get("Activity Alert", "").lower()
                hands = data.get("HOW Alert", "").lower()
                health = data.get("Health Alert", "").lower()
                distraction = data.get("Distraction Alert", "").lower()

                # Delete file after processing
                #os.remove(json_file_path)

                #highest priority
                if fatigue == "on" or sleep == True:
                    speak("[ALERT] Fatigue or Sleep Alert detected.")
                    alert()
            
                if action != "safe driving" and distraction == "on":
                    speak("please drive safely and focus on the road")
                if hands == "off_wheel": 
                    speak("Please keep your hands on the wheel")   
                if health == "on":
                    speak("Please take care as your heart rate is high take break if needed")
                if action =="eating":
                    speak("Please avoid eating while driving")
                if action =="drinking":
                    speak("Please avoid drinking while driving")
                if action =="talking on the phone" or action =="texting on the phone":
                    speak("Please avoid using phone while driving")
                    
        time.sleep(0.5)  # avoid tight loop
    
    eel.ShowHood()
    PassToLlm()

    # Restart monitoring loop after response
    start_monitoring()

@eel.expose
def start_monitoring():
    #using threads so the monitoring_loop() doesn't block the main thread
    thread = threading.Thread(target=monitoring_loop)
    thread.daemon = True
    thread.start()

def alert():

    speak("Are you okay? Can you hear me?")
    eel.ShowHood()
    response = check_up(timeout=3)

    if response is None:
        speak("Dangerous! No response from driver.")
        BuzzerSound()
        eel.ExitHood()
        return
    else:
        speak("Do you want to talk to me?")
        answer = check_up(timeout=3)
        if answer and "yes" in answer:
            PassToLlm()  # You must define this function elsewhere
        else:
            speak("Okay, I'm here if you need anything.")
            eel.ExitHood()


def check_up(timeout=3):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        eel.DisplayMessage("[Listening for response...]")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            response = recognizer.recognize_google(audio).lower()
            print(f"[Heard]: {response}")
            return response
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            return None


def BuzzerSound():
    music_path = "www//assets//audio//buzzer.wav"
    pygame.mixer.init()
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.play()