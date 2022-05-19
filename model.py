import tensorflow as tf

from stacks import Stacks

import tensorflow_probability as tfp

tfd = tfp.distributions


class Model(tf.keras.Model):
    """
    Implements model architecture
    """
    def __init__(self, k_mix=4, state_size=16):
        super().__init__()
        # K components for Gaussian Mixture
        self.K = max(2, k_mix)
        # hidden state size for RNNs
        self.state_size = state_size

        self.n_bins = None
        self.n_frames = None
        self.n_windows = None

        self.w_0 = tf.keras.layers.Dense(self.state_size, use_bias=False)  # see formula [7] in paper

        self.stacks = []
        for i in range(13):
            self.stacks.append(Stacks(state_size=self.state_size, k_mix=k_mix))

        # self.net_layer_1 = Stacks(state_size=self.state_size, k_mix=k_mix)
        # self.net_layer_2 = Stacks(state_size=self.state_size, k_mix=k_mix)

        self.dense = tf.keras.layers.Dense(self.K * 3)  # mu, scale, alpha

    @tf.function
    def train(self, X):
        odd_params = []
        for i, x_tier in enumerate(X):
            odd_params.append(self.stacks[i](x_tier['even']))
        return odd_params

    @tf.function
    def generate(self):
        # if i % 2 == 0:
        #     gaussian_even_time = self.stacks[i](tier)
        #     h_odd_freq = tf.stack([h_even_time, h_odd_time], axis=-2)  # (n_frames, 2, n_bins, 16)
        #     h_odd_freq = tf.reshape(h_odd_freq, [-1, h_odd_freq.shape[-1]])  # (2n_frames, n_bins, 16)
        # else:
        #     h_even_freq = self.stacks[i](h_odd_freq)
        #     h_odd_time = tf.stack([h_even_freq, h_odd_freq], axis=-1)  # (n_frames, n_bins, 2)
        #     h_odd_time = tf.reshape(h_odd_time, [h_odd_time.shape[-3], -1])  # (n_frames, 2n_bins)
        ...

    @tf.function
    def compute_loss(self, X, step, summary=False):
        odd_params = self.train(X)
        loss = 0
        for i, odd_p in enumerate(odd_params):
            mu, scale, alpha = tf.split(odd_p, 3, axis=-1)
            gaussian = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=tf.nn.softmax(alpha)),
                components_distribution=tfd.Normal(loc=mu, scale=tf.math.softplus(scale)))
            loss += -gaussian.log_prob(X[i]['odd']).reduce_mean()

        if summary:
            tf.summary.histogram('mu', mu, step=step)
            tf.summary.histogram('scale', scale, step=step)
            tf.summary.histogram('alpha', alpha, step=step)
            tf.summary.scalar('loss', loss, step=step)

        return loss