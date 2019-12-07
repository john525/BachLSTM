import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, vocab_size, output_size):
        super(Model, self)

        # Define hyperparameters

        # Define layers

        # Define optimizer
        model = tf.keras.Sequential()
        # TODO: fill out dimensions here
        model.add(tf.keras.layers.Embedding(input_dim = self.vocab_size, output_dim = self.output_size))
        model.add(tf.keras.layers.LSTM(units = 100))
        model.add(tf.keras.layers.LSTM(units = 100))
        model.add(tf.keras.layers.Dense(units = self.vocab_size, activation = "softmax"))
        self.model = model
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

    @tf.function
    def call(self, inputs):
        return self.model(inputs)

    def loss(self, probabilites, labels):
        return tf.reduce_sum(self.loss(probabilities, labels))
