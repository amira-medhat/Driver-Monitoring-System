import pygame
import eel #to use this python script with java script 


@eel.expose
def playAssistantSound():
    music_path = "www//assets//audio//start_sound.mp3"
    pygame.mixer.init()
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

@eel.expose
def playClickSound():
    music_path = "www//assets//audio//click_sound.wav"
    pygame.mixer.init()
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

