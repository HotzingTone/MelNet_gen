import tensorflow as tf
import glob


class DataSource:

    def __init__(self, path):
        self.files = glob.glob(f'{path}/*.wav')
        self.fft_size = None
        self.hop_size = None

    @tf.function
    def split(self, tier):
        f = int(tier.shape[-2] / 2)  # n_freq / 2
        tier = tf.reshape(tier, [-1, f, 2, 1])
        even_freq = tier[:, :, 0, :]
        odd_freq = tier[:, :, 1, :]
        tier = tf.reshape(odd_freq, [-1, 2, f, 1])
        even_time = tier[:, 0, :, :]
        odd_time = tier[:, 1, :, :]
        return {'even': even_freq, 'odd': odd_freq}, {'even': even_time, 'odd': odd_time}

    @tf.function  # ?
    def get_tiers(self, file):
        audio, _ = tf.audio.decode_wav(tf.io.read_file(file))
        audio = tf.squeeze(audio, axis=-1)
        spectrogram = tf.abs(tf.signal.stft(audio, self.fft_size, self.hop_size, pad_end=True))
        spectrogram = tf.signal.frame(spectrogram[:, :-1], 64, 64, axis=0, pad_end=True)
        tier = tf.reshape(spectrogram, [-1, 512, 1])  # todo Add back the last bin later
        X = []
        for i in range(6):
            pair_freq, pair_time = self.split(tier)
            tier = pair_time['odd']
            X.extend([pair_freq, pair_time])
        # 12 tiers:
        # (None, 256, 1)
        # (None/2, 256, 1)
        # (None/2, 128, 1)
        # (None/4, 128, 1)
        # (None/4, 64, 1)
        # (None/8, 64, 1)
        # (None/8, 32, 1)
        # (None/16, 32, 1)
        # (None/16, 16, 1)
        # (None/32, 16, 1)
        # (None/32, 8, 1)
        # (None/64, 8, 1)
        X.reverse()
        return X

    def get_data(self, eval_mode=False, fft_size=1024, hop_size=256):
        train_files = []
        for i, f in enumerate(self.files):
            if ((i % 10) != 9) == eval_mode:
                continue
            train_files.append(f)
        ds = tf.data.Dataset.from_tensor_slices(train_files)
        self.fft_size = fft_size
        self.hop_size = hop_size
        data = ds.map(self.get_tiers)
        return data

    # todo: Normalization
