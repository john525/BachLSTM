def unload_data(data, token_dict, path_to_midi_out=''):
    mt = MidiTrack(1)
    for msg in data:
        me = MidiEvent()
    if path_to_mid_out is None:
        s.show('midi')
    else:
        fp = s.write('midi', fp=path_to_mid_out)
