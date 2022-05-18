import tensorflow as tf
import glob


class DataSource:

    def __init__(self, path):
        self.files = tf.data.Dataset.from_tensor_slices(glob.glob(f'{path}/*.wav'))
        self.fft_size = None
        self.hop_size = None

    @tf.function
    def split_freq(self, windows):
        _, f, b = windows.shape
        windows = tf.reshape(windows, [-1, f, int(b / 2), 2])
        even = windows[..., 0]
        odd = windows[..., 1]
        return even, odd

    @tf.function
    def split_time(self, windows):
        _, f, b = windows.shape
        windows = tf.reshape(windows, [-1, int(f / 2), 2, b])
        even = windows[..., 0, :]
        odd = windows[..., 1, :]
        return even, odd

    @tf.function
    def get_tiers(self, file):
        audio, _ = tf.audio.decode_wav(tf.io.read_file(file))
        audio = tf.squeeze(audio, axis=-1)
        magnitude = tf.abs(tf.signal.stft(audio, self.fft_size, self.hop_size, pad_end=True))
        windows = tf.signal.frame(magnitude, 128, 64, axis=0, pad_end=True)
        # discards the last window and the last bin
        # shape (n_windows, 128_frames, 512_bins)  # todo Add back the last bin later
        odd = windows[:-1, :, :-1]
        tiers = []
        for i in range(6):
            even, odd = self.split_freq(odd)
            tiers.append(even)
            even, odd = self.split_time(odd)
            tiers.append(even)
        tiers.append(odd)
        # all tier shapes:
        # (None, 128, 256)
        # (None, 64, 256)
        # (None, 64, 128)
        # (None, 32, 128)
        # (None, 32, 64)
        # (None, 16, 64)
        # (None, 16, 32)
        # (None, 8, 32)
        # (None, 8, 16)
        # (None, 4, 16)
        # (None, 4, 8)
        # (None, 2, 8)
        # (None, 2, 8)
        return tiers

    def get_data(self, fft_size=1024, hop_size=256):
        self.fft_size = fft_size
        self.hop_size = hop_size
        data = self.files.map(self.get_tiers)
        return data

    # todo: Normalization
