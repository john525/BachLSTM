import numpy as np
import tensorflow as tf
import os
from music21 import midi as midi

# MAX_DURATION = 200
MAX_PITCH = 128
MAX_CHANNEL = 16
MAX_VELOCITY = 128 # TODO: check

def tokenize_event(evt, token_dict):
    one_hot = np.zeros((2, MAX_PITCH, MAX_CHANNEL, MAX_VELOCITY))
    msg_type = int(evt.type == 'NOTE_ON')
    idx = msg_type, int(evt.pitch), int(evt.channel), int(evt.velocity)
    if idx not in token_dict:
        token_dict[idx] = len(token_dict)
    return token_dict[idx]

def load_data(path_to_midi_files, all_data=True):
    """
    path_to_midi_files - path to the dataset containing .mid files
    returns - a list of note vectors, and a token dictionary
    """
    files = [fname for fname in os.listdir(path_to_midi_files) if fname not in 'sankey bwv988 bwv232 leipzig']
    num_tokens = 0

    songs = []
    token_dict = {}

    if all_data == False:
        files = files[:10]

    print()
    for i,fname in enumerate(files):
        print('\r\r Reading file %d out of %d...' % (i+1, len(files)))
        midi_file = midi.MidiFile()
        midi_file.open(os.path.join(path_to_midi_files, fname), 'rb')
        midi_file.read()
        midi_file.close()

        # TODO: normalize ticks per beat

        # Convert midi song to a vector
        song_vector = [] # TODO: represent each event as a one-hot, remove metadata events
        for channel in midi_file.tracks:
            for event in channel.events:
                if event.type in 'NOTE_ON NOTE_OFF':
                    song_vector.append(tokenize_event(event, token_dict))
        songs.append(song_vector)

    print('\nUnique Tokens: %d' % len(token_dict))
    for i in range(len(songs)):
        songs[i] = tf.one_hot(songs[i], len(token_dict))
        
    return songs, {token_dict[data]: data for data in token_dict}
