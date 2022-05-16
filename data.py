import tensorflow as tf
import glob


def create_dataset(path, batch_size=8, fft_size=1024, hop_size=256):
    def map_func(file) -> tf.Tensor:
        audio, _ = tf.audio.decode_wav(tf.io.read_file(file))
        audio = tf.squeeze(audio, axis=-1)
        magnitude = tf.abs(tf.signal.stft(audio, fft_size, hop_size, pad_end=True))
        samples = tf.signal.frame(magnitude, 128, 64, axis=0)  # (n_samples, 128_frames, 513_bins)
        return tf.data.Dataset.from_tensor_slices(samples)
    files = tf.data.Dataset.from_tensor_slices(glob.glob(f'{path}/*.wav'))
    dataset = files.flat_map(map_func).batch(batch_size)  # (n_batches, 8, 128, 513)
    return dataset
