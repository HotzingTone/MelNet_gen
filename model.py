import tensorflow as tf

from stacks import Stack

import tensorflow_probability as tfp

tfd = tfp.distributions


class Model(tf.keras.Model):

    def __init__(self, k_mix=4, h_size=16):
        super().__init__()
        # K components for Gaussian Mixture
        self.K = max(2, k_mix)
        # hidden state size for RNNs
        self.h_size = h_size
        # units for bottom tier generator
        self.init_x = tf.zeros([1, 1, 8])
        self.init_h = tf.zeros([1, self.h_size * 8])
        self.bottom_gru = tf.keras.layers.GRU(self.h_size * 8)
        self.bottom_dense = tf.keras.layers.Dense(self.K * 3 * 8)
        # all stack tiers
        self.stack_tiers = []
        for i in range(12):
            self.stack_tiers.append(Stack(h_size=self.h_size, k_mix=k_mix))

    @tf.function
    def bottom_generator(self, x, h):  # (1, 1, 8) (1, h_size)
        h = self.bottom_gru(x, initial_state=h)
        p = self.bottom_dense(h)  # gaussian params
        p = tf.reshape(p, [1, 8, -1])  # (1, 8, 3K)
        return p, h

    # @tf.function  # ?
    def train(self, X):
        odd_params = []
        for i, tier in enumerate(X):
            even_x = tier['even']
            odd_params.append(self.stack_tiers[i](even_x))  # (n_frames, n_bins, 3K)
            # bottom tier sequential generator training
            if i == 0:
                # bottom tier, one frame per rnn step, (n_frames, 1, 1, 8)
                bottom_x = tf.reshape(even_x, [-1, 1, 1, 8])
                # shift ground true x one frame forward for teacher-forcing, (n_frames, 1, 1, 8)
                bottom_x = tf.concat([tf.expand_dims(self.init_x, 0), bottom_x[:-1, ...]], axis=0)
                # initial hidden state
                h = self.init_h
                # gaussian params for bottom tier
                bottom_p = []
                for x in bottom_x:
                    p, h = self.bottom_generator(x, h)  # (1, 1, 8) (1, h_size) -> (1, 8, 3K) (1, h_size)
                    bottom_p.append(p)
                bottom_p = tf.concat(bottom_p, axis=0)  # (n_frames, 8, 3K)

        return bottom_p, odd_params

    @tf.function  # ?
    def generate(self, length):
        x = self.init_x
        h = self.init_h
        for i in range(length):
            p, h = self.bottom_generator(x, h)  # (1, 1, 8) (1, h_size) -> (1, 8, 3K) (1, h_size)
            gmx = self.gaussian_mix(p)  # batch_shape (1, 8), event_shape (,)
            x = gmx.sample()  # (1, 8)

        # if i % 2 == 0:
        #     gaussian_even_time = self.stacks[i](tier)
        #     h_odd_freq = tf.stack([h_even_time, h_odd_time], axis=-2)  # (n_frames, 2, n_bins, h_size)
        #     h_odd_freq = tf.reshape(h_odd_freq, [-1, h_odd_freq.shape[-1]])  # (2n_frames, n_bins, h_size)
        # else:
        #     h_even_freq = self.stacks[i](h_odd_freq)
        #     h_odd_time = tf.stack([h_even_freq, h_odd_freq], axis=-1)  # (n_frames, n_bins, 2)
        #     h_odd_time = tf.reshape(h_odd_time, [h_odd_time.shape[-3], -1])  # (n_frames, 2n_bins)
        ...

    # @tf.function(experimental_relax_shapes=True) ?
    def gaussian_mix(self, params):
        mu, scale, alpha = tf.split(params, 3, axis=-1)
        gmx = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=tf.nn.softmax(alpha)),
            components_distribution=tfd.Normal(loc=mu, scale=tf.math.softplus(scale)))
        return gmx

    # @tf.function  # ?
    def compute_loss(self, X, step, summary=False):
        # get gaussian mixture params for each tier
        bottom_p, odd_params = self.train(X)  # (n_frames, 8, 3K), list of 12 (n_frames, 8, 3K)
        # compute negative log-likelihood for each tier
        loss = 0
        for i, odd_p in enumerate(odd_params):
            if i == 0:
                bottom_x = X[0]['even']  # (n_frames, 8, 1)
                gmx = self.gaussian_mix(bottom_p)  # gmx batch_shape (n_frames, 8), event_shape (,)
                nll = -gmx.log_prob(tf.squeeze(bottom_x, -1))  # (n_frames, 8))
                loss += tf.reduce_mean(nll) / 13
            odd_x = X[i]['odd']  # (n_frames, 8, 1)
            gmx = self.gaussian_mix(odd_p)  # batch_shape (n_frames, 8), event_shape (,)
            nll = -gmx.log_prob(tf.squeeze(odd_x, -1))  # (n_frames, 8)
            loss += tf.reduce_mean(nll) / 13
        if summary:
            tf.summary.scalar('loss', loss, step=step)
        return loss
