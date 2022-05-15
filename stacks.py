import tensorflow as tf


class Stacks(tf.keras.layers.Layer):
    """
    Time-delayed stack -> Centralized stack -> Frequency-delayed stack
    """
    def __init__(self, state_size=32, k_mix=4, mode=None):
        super().__init__()
        self.mode = mode
        self.state_size = state_size  # small size 16 for demo
        self.K = k_mix  # K components for Gaussian Mixture
        # default MelNet mode, initiate GRU units and W matrices for all stacks
        # if set to Baseline, only Centralized stack is initiated
        if self.mode != 'baseline':
            # time-delayed units
            self.gru_time_forth = tf.keras.layers.GRU(self.state_size, return_sequences=True, time_major=True)
            self.gru_time_up = tf.keras.layers.GRU(self.state_size, return_sequences=True)
            self.gru_time_down = tf.keras.layers.GRU(self.state_size, return_sequences=True, go_backwards=True)
            # self.wt_0 = tf.keras.layers.Dense(self.state_size, use_bias=False)  # see formula [7] in paper
            self.Wt = tf.keras.layers.Dense(self.state_size, use_bias=False)  # see formula [6] in paper
            # frequency-delay units
            self.gru_frequency = tf.keras.layers.GRU(self.state_size, return_sequences=True)
            # self.wf_0 = tf.keras.layers.Dense(self.state_size, use_bias=False)  # see formula [9] in paper
            self.Wf = tf.keras.layers.Dense(self.state_size, use_bias=False)  # see formula [8] in paper
        # centralized units
        self.gru_centralize = tf.keras.layers.GRU(self.state_size, return_sequences=True)
        # self.wc_0 = tf.keras.layers.Dense(self.state_size, use_bias=False)  # see formula [12] in paper
        self.Wc = tf.keras.layers.Dense(self.state_size, use_bias=False)  # see formula [11] in paper
        # final linear layer
        self.dense = None

    def time_delayed(self, h_t):
        # run 3 RNNs in differect directions
        RNN_forth = self.gru_time_forth(h_t)
        RNN_up = self.gru_time_up(h_t)
        RNN_down = self.gru_time_down(h_t)
        # concatenate hidden states of 3 RNNs as input to residual block
        RNNs = tf.concat([RNN_forth, RNN_up, RNN_down], axis=-1)
        # residual block, see formula [6], output shape (128, 80, 16)
        Ht = self.Wt(RNNs) + h_t
        return Ht  # shape (128, 80, 16)

    def centralized(self, h_c):
        # residual block, see formula [11], output shape (1, 128, 16)
        Hc = self.Wc(self.gru_centralize(h_c)) + h_c
        return Hc

    def frequency_delayed(self, h_f, Ht, Hc):
        # residual block, see formula [8], output shape (128, 80, 16)
        Hf = self.Wf(self.gru_frequency(h_f + Ht + Hc)) + h_f
        return Hf  # shape (128, 80, 16)

    def call(self, h_c, h_t=None, h_f=None):  # input shapes (128, 80, 1)
        # Baseline mode, only using Centrailzed stack with a linear layer.
        if self.mode == 'baseline':
            Hc = self.centralized(h_c)  # shape (1, 128, 16)
            return Hc
        # MelNet mode, using all 3 stacks and the final linear layer.
        else:
            Ht = self.time_delayed(h_t)  # shape (128, 80, 16)
            Hc = self.centralized(h_c)  # shape (1, 128, 16)
            Hf = self.frequency_delayed(h_f, Ht, tf.squeeze(tf.expand_dims(Hc, -2), 0))  # (128, 80, 16)
            return Hc, Ht, Hf
