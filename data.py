import tensorflow as tf
import glob


class Data:

    def __init__(self, path):
        self.files = tf.data.Dataset.from_tensor_slices(glob.glob(f'{path}/*.wav'))

    def create_dataset(self, fft_size=1024, hop_size=256):
        @tf.function
        def get_spectrogram(file):
            audio, _ = tf.audio.decode_wav(tf.io.read_file(file))
            audio = tf.squeeze(audio, axis=-1)
            magnitude = tf.abs(tf.signal.stft(audio, fft_size, hop_size, pad_end=True))
            spectrogram = tf.signal.frame(magnitude, 128, 64, axis=0, pad_end=True)
            # discards the last windon and the last bin
            # shape (n_windows, 128_frames, 512_bins)
            spectrogram = spectrogram[:-1, :, :-1]
            return spectrogram
        dataset = self.files.map(get_spectrogram)
        return dataset

    # todo: Normalization
