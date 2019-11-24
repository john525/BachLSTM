import numpy as np
import tensorflow as tf
import os
import mido

MAX_NOTE = 128
MAX_CHANNEL = 16
MAX_NOTE_LEN = 600 # TODO: check
MAX_VELOCITY = 128 # TODO: check

def find_end_of_note(midi_file, note, channel, time):
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
    files = os.listdir(path_to_midi_files)
    num_tokens = 0
    vector_tracks = []

    word_limit = 1.6e6
    if all_data == False:
        word_limit = 1.6e3

    for i,fname in enumerate(files):
        print('loading file %d out of %d...' % (i+1, len(files)))
        vector_track = []
        midi_file = mido.MidiFile(os.path.join(path_to_midi_files, fname))

        # Normalize ticks per beat, so that maximum note length in ticks=100
        # ticks_per_beat = midi_file.ticks_per_beat
        # max_len = 0
        # for msg in midi_file:
        #     if msg.type == 'note_on':
        #         end_time = find_end_of_note(midi_file, msg.note, msg.channel, msg.time)
        #         length = end_time - msg.time
        #         if length > max_len:
        #             max_len = length
        # TODO: how to normalize? We need to somehow set max_len to MAX_NOTE_LEN

        # Combine every note_on and note_off into a single token, then concatenate based on start time
        num_notes = len([msg for msg in midi_file if not msg.is_meta])
        for chan in range(MAX_CHANNEL):
            for j,msg in enumerate(midi_file):
                if (j+1) % 200 == 0:
                    print('loading note %d out of %d...' % (j+1, num_notes))
                if msg.is_meta or msg.channel != chan:
                    continue
                if msg.type == 'note_on':
                    note = msg.note
                    velocity = msg.velocity
                    start_time = msg.time
                    channel = msg.channel
                    end_time = find_end_of_note(midi_file, note, channel, start_time)
                    vec = note_to_vec(note, channel, end_time - start_time, velocity)
                    vector_track.append(vec)
        vector_tracks.append(vector_track)
        num_words += len(vector_track)

        if num_words >= word_limit:
            break # Huang and Wu's dataset had 1.6 million words

    return vector_tracks
