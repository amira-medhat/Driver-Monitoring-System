############### chrome browser ###############
# import eel
# import os
# # Initialize Eel with your frontend folder
# eel.init("www")

# # Start Eel WITHOUT opening the browser (mode=None)
# os.system('start msedge.exe --app="http://localhost:8000/index.html"')
# eel.start('index.html', mode=None, host='localhost', block=True)


############### firefox browser ###############
import eel
import webbrowser

from engine.features import *
from engine.command import *
# Initialize Eel with your frontend folder
eel.init("www")

playAssistantSound()



# Open Firefox in app mode or normal window (adjust this line as needed)
webbrowser.get("firefox").open_new("http://localhost:8000/index.html")

threading.Thread(target=monitoring_loop, daemon=True).start()

# Start the Eel server (without trying to open a browser)
eel.start('index.html', mode=None, host='localhost', port=8000, block=True)


