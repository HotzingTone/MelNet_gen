import tensorflow as tf

from stacks import Stacks

import tensorflow_probability as tfp

tfd = tfp.distributions


class Model(tf.keras.Model):
    """
    Implements model architecture
    """
    def __init__(self, k_mix=4, state_size=32, n_bins=None, mode=None):
        super().__init__()
        # K components for Gaussian Mixture
        self.K = k_mix
        # hidden state size for RNNs
        self.state_size = state_size
        # number of log-mel bins
        self.n_bins = n_bins
        self.n_frames = None
        self.n_samples = None
        # optional Baseline mode or default MelNet mode
        self.mode = mode
        self.wc_0 = tf.keras.layers.Dense(self.state_size, use_bias=False)  # see formula [12] in paper
        if self.mode == 'baseline':
            print('\nUsing Baseline model...')
            self.net_baseline = Stacks(state_size=self.state_size, k_mix=1, mode='baseline')
            self.dense = tf.keras.layers.Dense(self.n_bins * 2)  # mu, scale
        else:
            self.wt_0 = tf.keras.layers.Dense(self.state_size, use_bias=False)  # see formula [7] in paper
            self.wf_0 = tf.keras.layers.Dense(self.state_size, use_bias=False)  # see formula [9] in paper
            print('\nUsing MelNet model...')
            self.net_layer_1 = Stacks(state_size=self.state_size, k_mix=k_mix)
            self.net_layer_2 = Stacks(state_size=self.state_size, k_mix=k_mix)
            if self.K == 1:
                self.dense = tf.keras.layers.Dense(2)  # mu, scale
            else:
                self.dense = tf.keras.layers.Dense(self.K * 3)  # mu, scale, alpha

    @tf.function
    def call(self, inputs, targets):
        self.n_samples, self.n_frames, self.n_bins = inputs.shape  # (8, 128, 80)

        # x_c take a whole frame as one step of input for RNN,
        # used as inputs for centralized stack
        # project x to h, see formula [12], output shape (8, 1, 128, 16)
        x_c = tf.reshape(inputs, [self.n_samples, 1, self.n_frames, self.n_bins])
        h_c = self.wc_0(x_c)

        if self.mode == 'baseline':
            # outputs are parameters for multivariate Gaussian
            # shape (8, 128, 80, 2)
            outputs = tf.stack([self.stack_layers(h_c[i])
                                for i in range(self.n_samples)], axis=0)  # (8, 128, 80, 2)
            return outputs
        else:
            # x_t is one frame earlier than targets,
            # used as inputs for time-delayed stack
            # project x to h, see formula [7], output shape (8, 128, 80, 16)
            x_t = tf.expand_dims(inputs, axis=-1)
            h_t = self.wt_0(x_t)
            # x_f is one log-mel bin lower than targets,
            # used as input for frequency-delayed stack,
            # project x to h, see formula [9], output shape (8, 128, 80, 16)
            x_f = tf.concat([tf.zeros([self.n_samples, self.n_frames, 1]), targets[:, :, :-1]], axis=-1)
            x_f = tf.expand_dims(x_f, axis=-1)
            h_f = self.wf_0(x_f)
            # outputs are parameters for Gaussian Mixture with K components
            # shape (8, 128, 80, 3K)
            outputs = tf.stack([self.stack_layers(h_c[i], h_t[i], h_f[i])
                                for i in range(self.n_samples)], axis=0)  # (8, 128, 80, 3K)
            return outputs

    @tf.function
    def stack_layers(self, h_c, h_t=None, h_f=None):
        if self.mode == 'baseline':
            h_c = self.net_baseline(h_c)
            outputs = self.dense(h_c)
            outputs = tf.reshape(outputs, [self.n_frames, self.n_bins, -1])  # shape (128, 80, 2)
        else:
            # 2 layer networks consisting of Centralized / Time / Frequency stacks
            h_c, h_t, h_f = self.net_layer_1(h_c, h_t, h_f)
            h_c, h_t, h_f = self.net_layer_2(h_c, h_t, h_f)
            outputs = self.dense(h_f)  # shape (128, 80, 3K)
        return outputs

    @tf.function
    def loss_func(self, outputs, targets):
        if self.K == 1 or self.mode == 'baseline':
            mu, scale = tf.split(outputs, 2, axis=-1)
            mu = tf.squeeze(mu, -1)
            scale = tf.squeeze(scale, -1)
            dist = tfd.Normal(loc=mu, scale=tf.math.softplus(scale))
            loss = -dist.log_prob(targets)
        else:
            mu, scale, alpha = tf.split(outputs, 3, axis=-1)
            gaussian_mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=tf.nn.softmax(alpha)),
                components_distribution=tfd.Normal(loc=mu, scale=tf.math.softplus(scale)))
            loss = -gaussian_mix.log_prob(targets)
        loss = tf.reduce_mean(loss)
        return loss

    @tf.function
    def compute_loss(self, inputs, targets):
        outputs = self.call(inputs, targets)  # (8, 128, 80) -> (8, 128, 80, 3K)
        return self.loss_func(outputs, targets)

    @tf.function
    def compute_loss_with_summaries(self, inputs, targets, global_step):
        outputs = self.call(inputs, targets)  # (8, 128, 80) -> (8, 128, 80, 3K)
        if self.K == 1 or self.mode == 'baseline':
            mu, scale = tf.split(outputs, 2, axis=-1)
            tf.summary.histogram('mu', mu, step=global_step)
            tf.summary.histogram('scale', scale, step=global_step)
        else:
            mu, scale, alpha = tf.split(outputs, 3, axis=-1)
            tf.summary.histogram('mu', mu, step=global_step)
            tf.summary.histogram('scale', scale, step=global_step)
            tf.summary.histogram('alpha', alpha, step=global_step)
        return self.loss_func(outputs, targets)