import tensorflow as tf


class Stacks(tf.keras.layers.Layer):
    """
    Time-delayed stack -> Centralized stack -> Frequency-delayed stack
    """
    def __init__(self, state_size=16, k_mix=4):
        super().__init__()
        self.state_size = state_size  # small size 16 for demo
        self.K = k_mix  # K components for Gaussian Mixture
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
        self.wt_0 = tf.keras.layers.Dense(
            self.state_size, use_bias=False  # see formula [7] in paper
        )
        self.Wt = tf.keras.layers.Dense(
            self.state_size, use_bias=False  # see formula [6] in paper
        )
        # final linear layer
        self.dense = tf.keras.layers.Dense(self.K * 3)

    def call(self, x_tier):
        h_tier = self.wt_0(x_tier)
        # run 4 RNNs in different directions
        RNN_forth = self.gru_forth(h_tier)
        RNN_back = self.gru_back(h_tier)
        RNN_up = self.gru_up(h_tier)
        RNN_down = self.gru_down(h_tier)
        # concatenate hidden states of 4 RNNs as input to residual block
        RNNs = tf.concat([RNN_forth, RNN_back, RNN_up, RNN_down], axis=-1)
        # residual block, see formula [6], output shape (n_frames, n_bins, 16)
        p_tier = self.dense(self.Wt(RNNs) + h_tier)
        return p_tier
