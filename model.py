import tensorflow as tf
from tensorflow.keras import layers

def get_keras_model():
    model = tf.keras.Sequential()

    model.add(layers.Embedding(input_dim = 8326, output_dim = 8326))
    model.add(layers.LSTM(100, input_shape=(100,)))
    model.add(layers.Dense(8326, input_shape=(100,), activation = 'softmax'))

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer=optimizer, loss=loss)

    return model

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        # Define hyperparameters
        self.vocab_size = 8326
        self.output_size = 8326
        self.batch_size = 52

        # Define layers

        # Define optimizer
        # TODO: fill out dimensions here
        self.embedding = tf.keras.layers.Embedding(input_dim = self.vocab_size, output_dim = self.output_size)
        self.lstm1 = tf.keras.layers.LSTM(100, input_shape=(100,))
        # self.lstm2 = tf.keras.layers.LSTM(units = 100)
        self.dense = tf.keras.layers.Dense(self.vocab_size, input_shape=(100,), activation = 'softmax')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

    @tf.function
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm1(x)
        # x = self.lstm2(x)
        x = self.dense(x)
        return x

    @tf.function
    def loss(self, probabilities, labels):
        y_true = labels
        y_pred = probabilities
        losses = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return tf.reduce_sum(losses)
