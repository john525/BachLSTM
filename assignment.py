from model import Model
from preprocessing import MidiLoader
from postprocessing import unload_data
import sys
import tensorflow as tf
import datetime as datetime

def train(model, data, labels):
    start_time = datetime.datetime.now()
    for i in range(0, len(data), model.batch_size):
        start_idx = i
        end_idx = min(i + model.batch_size, len(data))

        if end_idx % model.batch_size != 0:
            return int((datetime.datetime.now() - start_time).seconds)

        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        with tf.GradientTape() as g:
            logits = model.call(batch_data)
            loss = model.loss(logits, batch_labels)

        seconds = int((datetime.datetime.now() - start_time).seconds)
        minutes = int(seconds / 60)
        seconds %= 60

        print(' (%02d:%02d) batch %03d/%03d, loss=%07.2F' % (minutes, seconds, i / model.batch_size + 1, len(data) / model.batch_size, loss))
        grad = g.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grad, model.trainable_variables))

    return int((datetime.datetime.now() - start_time).seconds)

def test(model, data, labels):
    total_loss = 0.0
    n = 0.0

    start_time = datetime.datetime.now()
    for i in range(0, len(data), model.batch_size):
        start_idx = i
        end_idx = min(i + model.batch_size, len(data))

        if end_idx % model.batch_size != 0:
            break

        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        logits = model.call(batch_data)
        loss = model.loss(logits, batch_labels)
        total_loss += loss
        n += 1.0
    print('Test Loss: %f' % total_loss)
    print('Test Perp: %f' % tf.exp(total_loss / n))

    seconds = int((datetime.datetime.now() - start_time).seconds)
    minutes = int(seconds / 60)
    seconds %= 60
    print('Test Time: %02d:%02d' % (minutes, seconds))

def main():
    small_dataset = False
    if len(sys.argv) != 2 or sys.argv[1] not in {"BIG", "SMALL"}:
        print("USAGE: python assignment.py <Dataset>")
        print("<Dataset>: [BIG/SMALL]")
        exit()

    print('=== Bach LSTM Generator (nwee, jlhota) ===')
    full_dataset = False
    midi_loader = MidiLoader()
    m = Model()

    if sys.argv[1] == "BIG":
        full_dataset = True
    elif sys.argv[1] == "SMALL":
        data, labels, token_dict = midi_loader.load_data('./data/jsbach.net/midi/', all_data=False)

    print('=== Training ===')
    if not full_dataset:
        all_songs_data = tf.concat(data, axis=0)
        all_songs_labels = tf.concat(labels, axis=0)
        test_train_cutoff = int(len(all_songs_data) * 0.9)
        train_data = all_songs_data[:test_train_cutoff]
        test_data = all_songs_data[test_train_cutoff:]
        train_labels = all_songs_labels[:test_train_cutoff]
        test_labels = all_songs_labels[test_train_cutoff:]

        train(m, train_data[:-1], train_labels[1:])
    else:
        total_time = 0
        while True:
            data, labels, token_dict = midi_loader.load_data('./data/jsbach.net/midi/', all_data=True)

            if data == None:
                break

            all_songs_data = tf.concat(data, axis=0)
            all_songs_labels = tf.concat(labels, axis=0)
            test_train_cutoff = int(len(all_songs_data) * 0.9)
            train_data = all_songs_data[:test_train_cutoff]
            test_data = all_songs_data[test_train_cutoff:]
            train_labels = all_songs_labels[:test_train_cutoff]
            test_labels = all_songs_labels[test_train_cutoff:]

            time = train(m, train_data[:-1], train_labels[1:])

            if time is None:
                print('break1')
                break
            total_time += time

            print('=== Testing ===')
            test(m, test_data[:-1], test_labels[1:])
            print()

            # if total_time >= 12*60*60:
            #     print('break2')
            #     break


    print('=== Testing ===')
    test(m, test_data[:-1], test_labels[1:])

if __name__ == '__main__':
    main()
