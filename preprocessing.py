import numpy as np
import tensorflow as tf
import os
from music21 import midi as midi

# MAX_DURATION = 200
MAX_PITCH = 128
MAX_CHANNEL = 16
MAX_VELOCITY = 128 # TODO: check

FILE_INTERVAL = 2

class MidiLoader:
    def __init__(self, f_index=0):
        self.file_index = f_index

        for i in range(f_index):
            self.load_data('./data/jsbach.net/midi')

    def tokenize_event(self, evt):
        b = evt.getBytes()
        if b not in self.token_dict:
            self.token_dict[b] = len(self.token_dict)
        return self.token_dict[b]

    def load_data(self, path_to_midi_files, all_data=True):
        """
        path_to_midi_files - path to the dataset containing .mid files
        returns - a list of note vectors, and a token dictionary
        """
        files = [fname for fname in os.listdir(path_to_midi_files) if fname not in 'sankey bwv988 bwv232 leipzig']
        num_tokens = 0

        songs = []
        self.token_dict = {}

        if all_data == False:
            files = files[:FILE_INTERVAL]
        elif self.file_index >= len(files):
            return None, None, None
        else:
            files = files[self.file_index : min(self.file_index + FILE_INTERVAL, len(files))]
            self.file_index += FILE_INTERVAL

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
                        song_vector.append(self.tokenize_event(event))
            songs.append(song_vector)

        print('\nUnique Tokens: %d' % len(self.token_dict))
        labels = []
        for i in range(len(songs)):
            labels.append([x for x in songs[i]])
            songs[i] = tf.one_hot(songs[i], len(self.token_dict))

        songs = tf.concat(songs, axis=0)
        labels = tf.concat(labels, axis=0)

        return songs, labels, {self.token_dict[data]: data for data in self.token_dict}

    def count_data_tokens(self, path_to_midi_files, all_data=True):
        """
        path_to_midi_files - path to the dataset containing .mid files
        returns - a list of note vectors, and a token dictionary
        """
        files = [fname for fname in os.listdir(path_to_midi_files) if fname not in 'sankey bwv988 bwv232 leipzig']
        num_tokens = 0

        songs = []
        self.token_dict = {}

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
                        if bytes not in self.token_dict:
                            self.token_dict[bytes] = 1
                        else:
                            self.token_dict[bytes] = 1 + self.token_dict[bytes]

        keys = list(self.token_dict.values())
        keys.sort()
        for k in keys:
            print(k)
