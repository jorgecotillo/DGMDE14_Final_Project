from playsound import playsound; import os

def play(is_play,filepath):
    '''Spec - Plays MP3 sound
    Input - Bool --> is_play (True,False)|| String --> Complete file path '''
    if is_play and filepath.lower().endswith('.mp3'):
        playsound(filepath)
    else:
        print('Wrong file format at sound play')

is_play = True
filepath = os.path.join('C:\\Users\\nanda\\Documents\\harvard\\DGMDE14\\Final_Project','Wake Up'+'.mp3')
play(is_play, filepath)