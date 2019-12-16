from model import get_keras_model, loss_func
from preprocessing import MidiLoader
from postprocessing import unload_data
import sys
import tensorflow as tf
import datetime as datetime

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3 or sys.argv[1] not in {"BIG", "SMALL"}:
        print("USAGE: python assignment.py <n>")
        print("<n>: Start reading at file 2n")
        exit()

    print('=== Bach LSTM Generator (nwee, jlhota) ===')
    midi_loader = MidiLoader()
    m = get_keras_model()

    full_dataset = True
    if len(sys.argv) == 3:
        midi_loader = MidiLoader(int(sys.argv[2]))

    print('=== Training ===')

    if True:
        total_time = 0
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True)

        batch_idx = 0
        while True:
            data, labels, token_dict = midi_loader.load_data('./data/jsbach.net/midi/', all_data=True)

            if data is None:
                break

            test_train_cutoff = int(data.shape[0] * 0.9)
            train_data = data[:test_train_cutoff ]
            test_data = data[test_train_cutoff:]
            train_labels = labels[:test_train_cutoff]
            test_labels = labels[test_train_cutoff:]

            def round_to_128(x):
                new_len = len(x) - (len(x) % 128)
                return x[:new_len + 1]

            train_data = round_to_128(train_data)
            test_data = round_to_128(test_data)
            train_labels = round_to_128(train_labels)
            test_labels = round_to_128(test_labels)

            if batch_idx % 2 == 0:
                m.evaluate(test_data[:-1], test_labels[1:], batch_size=128)
            batch_idx += 1

            m.fit(train_data[:-1], train_labels[1:], batch_size=128, epochs=1, validation_data=(test_data[:-1],\
                test_labels[1:]), callbacks=[checkpointer], shuffle=False)

            m.reset_states()

if __name__ == '__main__':
    main()
