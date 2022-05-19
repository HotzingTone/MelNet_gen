import tensorflow as tf
import glob


class DataSource:

    def __init__(self, path):
        self.files = tf.data.Dataset.from_tensor_slices(glob.glob(f'{path}/*.wav'))
        self.fft_size = None
        self.hop_size = None

    @tf.function
    def split(self, windows):
        _, f = windows.shape
        windows = tf.reshape(windows, [-1, int(f/2), 2])
        even_freq = windows[..., 0]
        odd_freq = windows[..., 1]
        odd_freq = tf.reshape(odd_freq, [-1, 2, int(f/2)])
        even_time = odd_freq[..., 0, :]
        odd_time = odd_freq[..., 1, :]
        return even_freq, even_time, odd_time

    @tf.function
    def get_tiers(self, file):
        audio, _ = tf.audio.decode_wav(tf.io.read_file(file))
        audio = tf.squeeze(audio, axis=-1)
        spectrogram = tf.abs(tf.signal.stft(audio, self.fft_size, self.hop_size, pad_end=True))
        spectrogram = tf.signal.frame(spectrogram[:, :-1], 64, 64, axis=0, pad_end=True)
        odd_time = tf.reshape(spectrogram, [-1, 512])  # todo Add back the last bin later
        tiers = []
        for i in range(6):
            even_freq, even_time, odd_time = self.split(odd_time)
            tiers.extend([even_freq, even_time])
        tiers.append(odd_time)
        # all tier shapes:
        # (None, 256)
        # (None/2, 256)
        # (None/2, 128)
        # (None/4, 128)
        # (None/4, 64)
        # (None/8, 64)
        # (None/8, 32)
        # (None/16, 32)
        # (None/16, 16)
        # (None/32, 16)
        # (None/32, 8)
        # (None/64, 8)
        # (None/64, 8)
        return tiers

    def get_data(self, fft_size=1024, hop_size=256):
        self.fft_size = fft_size
        self.hop_size = hop_size
        data = self.files.map(self.get_tiers)
        return data

    # todo: Normalization
