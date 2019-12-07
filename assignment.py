from model import Model
from preprocessing import load_data
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
            return

        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        with tf.GradientTape() as tape:
            logits = model.call(batch_data)
            loss = model.loss(logits, batch_labels)

        seconds = int((datetime.datetime.now() - start_time).seconds)
        minutes = int(seconds / 60)
        seconds %= 60

        print('batch %02d/%02d: loss=%07.2F, t=%d:%d' % (i / model.batch_size + 1, len(data) / model.batch_size, loss, minutes, seconds))
        grad = g.gradient(loss, model.trainable_variables)
        model.opt.apply_gradients(zip(grad, model.trainable_variables))

def test(model, data, labels):
    total_loss = 0.0
    n = 0.0
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


def main():
    small_dataset = False
    if len(sys.argv) != 2 or sys.argv[1] not in {"BIG", "SMALL"}:
        print("USAGE: python assignment.py <Dataset>")
        print("<Dataset>: [BIG/SMALL]")
        exit()

    print('=== Bach LSTM Generator (nwee, jlhota) ===')
    if sys.argv[1] == "BIG":
        data, token_dict = load_data('./data/jsbach.net/midi/', all_data=True)
    elif sys.argv[1] == "SMALL":
        data, token_dict = load_data('./data/jsbach.net/midi/', all_data=False)

    m = Model(len(token_dict))

    all_songs_data = tf.concat(data, axis=0)
    test_train_cutoff = int(len(all_songs_data) * 0.9)
    train_data = all_songs_data[:test_train_cutoff]
    test_data = all_songs_data[test_train_cutoff:]

    print('=== Training ===')
    train(m, train_data[:-1], train_data[1:])

    print('=== Testing ===')
    test(m, test_data[:-1], test_data[1:])

if __name__ == '__main__':
    main()
