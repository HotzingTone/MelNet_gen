import tensorflow as tf


class Stack(tf.keras.layers.Layer):
    def __init__(self, h_size=16, k_mix=4):
        super().__init__()
        self.h_size = h_size
        self.K = k_mix

        self.gru_forth = tf.keras.layers.GRU(
            self.h_size, return_sequences=True, time_major=True
        )
        self.gru_back = tf.keras.layers.GRU(
            self.h_size, return_sequences=True, time_major=True, go_backwards=True
        )
        self.gru_up = tf.keras.layers.GRU(
            self.h_size, return_sequences=True
        )
        self.gru_down = tf.keras.layers.GRU(
            self.h_size, return_sequences=True, go_backwards=True
        )
        self.wt_0 = tf.keras.layers.Dense(
            self.h_size, use_bias=False  # see formula [7] in paper
        )
        self.Wt = tf.keras.layers.Dense(
            self.h_size, use_bias=False  # see formula [6] in paper
        )
        self.dense = tf.keras.layers.Dense(self.K * 3)

    @tf.function
    def call(self, even_x):
        even_h = self.wt_0(even_x)
        # run 4 RNNs in different directions
        RNN_forth = self.gru_forth(even_h)
        RNN_back = self.gru_back(even_h)
        RNN_up = self.gru_up(even_h)
        RNN_down = self.gru_down(even_h)
        # concatenate hidden states of 4 RNNs as input to residual block
        RNNs = tf.concat([RNN_forth, RNN_back, RNN_up, RNN_down], axis=-1)
        # residual block, see formula [6], output shape (n_frames, n_bins, h_size)
        odd_h = self.Wt(RNNs) + even_h
        # gaussian parameters for
        odd_p = self.dense(odd_h)
        return odd_p
