import numpy as np
import tensorflow as tf
import os
from music21 import midi as midi

# MAX_DURATION = 200
MAX_PITCH = 128
MAX_CHANNEL = 16
MAX_VELOCITY = 128 # TODO: check

class MidiLoader:
    def __init__(self):
        self.file_index = 0

    def tokenize_event(self, evt, token_dict):
        b = evt.getBytes()
        if b not in token_dict:
            token_dict[b] = len(token_dict)
        return token_dict[b]

    def load_data(self, path_to_midi_files, all_data=True):
        """
        path_to_midi_files - path to the dataset containing .mid files
        returns - a list of note vectors, and a token dictionary
        """
        files = [fname for fname in os.listdir(path_to_midi_files) if fname not in 'sankey bwv988 bwv232 leipzig']
        num_tokens = 0

        songs = []
        token_dict = {}

        if all_data == False:
            files = files[:5]
        elif self.file_index >= len(files):
            return None, None, None
        else:
            files = files[self.file_index : min(self.file_index + 5, len(files))]
            self.file_index += 5

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
                        song_vector.append(self.tokenize_event(event, token_dict))
            songs.append(song_vector)

        print('\nUnique Tokens: %d' % len(token_dict))
        labels = []
        for i in range(len(songs)):
            labels.append([x for x in songs[i]])
            songs[i] = tf.one_hot(songs[i], len(token_dict))

        songs = tf.concat(songs, axis=0)
        labels = tf.concat(labels, axis=0)

        return songs, labels, {token_dict[data]: data for data in token_dict}

    def count_data_tokens(self, path_to_midi_files, all_data=True):
        """
        path_to_midi_files - path to the dataset containing .mid files
        returns - a list of note vectors, and a token dictionary
        """
        files = [fname for fname in os.listdir(path_to_midi_files) if fname not in 'sankey bwv988 bwv232 leipzig']
        num_tokens = 0

        songs = []
        token_dict = {}

        if all_data == False:
            files = files[:5]
        elif self.file_index >= len(files):
            return None, None, None
        else:
            files = files[self.file_index : min(self.file_index + 5, len(files))]
            self.file_index += 5

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
                        bytes = event.getBytes()
                        if bytes not in token_dict:
                            token_dict[bytes] = 1
                        else:
                            token_dict[bytes] = 1 + token_dict[bytes]

        keys = list(token_dict.values())
        keys.sort()
        for k in keys:
            print(k)
