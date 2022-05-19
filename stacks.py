import tensorflow as tf


class Stacks(tf.keras.layers.Layer):
    """
    Time-delayed stack -> Centralized stack -> Frequency-delayed stack
    """
    def __init__(self, state_size=16, k_mix=4):
        super().__init__()
        self.state_size = state_size  # small size 16 for demo
        self.K = k_mix  # K components for Gaussian Mixture
        # default MelNet mode, initiate GRU units and W matrices for all stacks
        # if set to Baseline, only Centralized stack is initiated

        # time-delayed units
        self.gru_forth = tf.keras.layers.GRU(
            self.state_size, return_sequences=True, time_major=True
        )
        self.gru_back = tf.keras.layers.GRU(
            self.state_size, return_sequences=True, time_major=True, go_backwards=True
        )
        self.gru_up = tf.keras.layers.GRU(
            self.state_size, return_sequences=True
        )
        self.gru_down = tf.keras.layers.GRU(
            self.state_size, return_sequences=True, go_backwards=True
        )
        # self.wt_0 = tf.keras.layers.Dense(self.state_size, use_bias=False)  # see formula [7] in paper
        self.Wt = tf.keras.layers.Dense(
            self.state_size, use_bias=False
        )  # see formula [6] in paper
        # final linear layer
        self.dense = None

    def time_delayed(self, h_t):
        # run 3 RNNs in different directions
        RNN_forth = self.gru_forth(h_t)
        RNN_back = self.gru_back(h_t)
        RNN_up = self.gru_up(h_t)
        RNN_down = self.gru_down(h_t)
        # concatenate hidden states of 3 RNNs as input to residual block
        RNNs = tf.concat([RNN_forth, RNN_back, RNN_up, RNN_down], axis=-1)
        # residual block, see formula [6], output shape (128, 80, 16)
        Ht = self.Wt(RNNs) + h_t
        return Ht  # shape (128, 80, 16)

    def call(self, h_c, h_t=None, h_f=None):  # input shapes (128, 80, 1)

        Ht = self.time_delayed(h_t)  # shape (128, 80, 16)
        return Ht
