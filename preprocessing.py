import numpy as np
import tensorflow as tf
import os
from music21 import midi as midi

MAX_NOTE = 128
MAX_CHANNEL = 16
MAX_NOTE_LEN = 600 # TODO: check
MAX_VELOCITY = 128 # TODO: check

def find_note_end(note_offs, note, channel, time):
    for msg in midi_file:
        if msg.type is 'note_off' or (msg.type is 'note_on' and msg.velocity == 0):
            if msg.time > time and msg.channel == channel and msg.note == note:
                return msg.time
    return midi_file.length

def note_to_vec(note, channel, length, velocity):
    one_hot = np.zeros((MAX_NOTE, MAX_CHANNEL, MAX_NOTE_LEN, MAX_VELOCITY))
    idx = int(note), int(channel), int(length), int(velocity)
    one_hot[idx] = 1.0
    return one_hot

def load_data(path_to_midi_files, all_data=True):
    """
    path_to_midi_files - path to the dataset containing .mid files
    returns - a list of note vectors
    """
    files = [fname for fname in os.listdir(path_to_midi_files) if fname not in 'sankey bwv988 bwv232 leipzig']
    num_tokens = 0
    vector_tracks = []

    word_limit = 1.6e6
    if all_data == False:
        word_limit = 1.6e3
    num_words = 0

    for i,fname in enumerate(files):
        vector_track = []

        midi_file = midi.MidiFile()
        midi_file.open(filename=os.path.join(path_to_midi_files, fname))

        # TODO: how to ticks per beat?

        # Combine every note_on and note_off into a single token, then concatenate based on start time
        for channel in midi_file.tracks:
            notes = channel.get_piano_roll(fs=30)
            print(notes)
        vector_tracks.append(vector_track)
        num_words += len(vector_track)

        if num_words >= word_limit:
            break # Huang and Wu's dataset had 1.6 million words

    return vector_tracks
