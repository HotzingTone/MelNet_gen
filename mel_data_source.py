import os

import tensorflow as tf
import glob

def search_for_wav_files(path) -> (list, int):
  wav_files = glob.glob(f'{path}/*.wav')
  if len(wav_files) == 0:
    raise Exception(f'Could not find any wav file in {path}')
  _, sample_rate = tf.audio.decode_wav(tf.io.read_file(wav_files[0]))
  return wav_files, int(sample_rate)


class PcmToMel(tf.keras.layers.Layer):
  '''Converts audio samples into mel spectrograms.'''

  def __init__(self, sample_rate, **kwargs):
    super().__init__(**kwargs)
    self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(80, 513, sample_rate, 80.0, 7600.0)

  def output_dim(self):
    return 80

  def call(self, pcm):
    stfts = tf.signal.stft(pcm, frame_length=1024, frame_step=256, fft_length=1024)
    spectrograms = tf.abs(stfts)  # [n_frames, 513_stft_bins]
    mel_spectrograms = tf.tensordot(spectrograms, self.linear_to_mel_weight_matrix, 1)  # [n_frames, 80_mel_bins]
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)  # [n_frames, 80_mel_bins]
    return log_mel_spectrograms[:, :self.output_dim()]  # [n_frames, n_mel_bins]


class MelDataSource(object):
  '''Load wav files as spectrograms from disk.'''

  def __init__(self, path):
    self.path = path
    self.wav_files, self.sample_rate = search_for_wav_files(path)
    self.pcm_to_mel = PcmToMel(self.sample_rate)
    self.mel_dim = self.pcm_to_mel.output_dim()
    print(f'Have {len(self.wav_files)} wav files with sample rate {self.sample_rate}')
    self.mel_stats_means = None
    self.mel_stats_stds = None
    self.load_or_build_stats()
    print('MEAN', self.mel_stats_means)
    print('STD', self.mel_stats_stds)

  def load_wav(self, path) -> (tf.Tensor, int):
    signal, sample_rate = tf.audio.decode_wav(tf.io.read_file(path))
    signal = tf.squeeze(signal, axis=-1)
    tf.assert_equal(sample_rate, self.sample_rate)
    return signal

  def create_dataset(self, batch_size: int = 8, eval_mode: bool = False, frame_length: int = 128, frame_step: int = 64):
    filenames = []
    for o, fn in enumerate(self.wav_files):
      if ((o % 10) != 9) == eval_mode:
        continue
      filenames.append(fn)
    ds = tf.data.Dataset.from_tensor_slices(filenames)

    @tf.function
    def map_func(path: tf.Tensor) -> tf.Tensor:
      pcm = self.load_wav(path)
      mels = self.pcm_to_mel(pcm)  # [n_frames, n_bins]
      mels -= tf.expand_dims(self.mel_stats_means, axis=0)  # mean normalised
      mels /= tf.expand_dims(self.mel_stats_stds, axis=0)  # std normalised
      inputs = mels[0:-1]  # [n_frames-1, n_bins]
      outputs = mels[1:]  # [n_frames-1, n_bins], 1 frame forward as ground truth
      input_frames = tf.signal.frame(inputs, frame_length, frame_step, axis=0)  # [..., 128_frames, n_bins], 64 overlap frames
      output_frames = tf.signal.frame(outputs, frame_length, frame_step, axis=0)  # [..., 128_frames, n_bins], 64 overlap frames
      return tf.data.Dataset.from_tensor_slices({'inputs': input_frames, 'targets': output_frames})
    ds = ds.flat_map(map_func)  # {'inputs': [..., 128, n_bins], 'targets', [..., 128, n_bins]}, collect all frame pairs.
    ds = ds.batch(batch_size)  # {'inputs': [..., 8, 128, n_bins], 'targets', [..., 8, 128, n_bins]}, batch size 8
    return ds

  def load_or_build_stats(self):
    stats_fn = os.path.join(self.path, 'norm_stats.json')
    print('Norm stats filename:', stats_fn)
    if os.path.exists(stats_fn):
      with open(stats_fn, 'r') as fp:
        lines = [x.strip().split() for x in fp.readlines()[1:]]
        self.mel_stats_means = tf.constant([float(x[1]) for x in lines])
        self.mel_stats_stds = tf.constant([float(x[2]) for x in lines])
      if self.mel_stats_means.shape[0] == self.mel_dim:
        return
    print('Generating normalization stats ...')
    mel_stats_sums = tf.constant(0.0, shape=[self.mel_dim])
    mel_stats_n = tf.constant(0, dtype=tf.int32)
    for fn in self.wav_files:
      mels = self.pcm_to_mel(self.load_wav(fn))
      mel_stats_sums += tf.reduce_sum(mels, axis=0)
      mel_stats_n += mels.shape[0]
    self.mel_stats_means = mel_stats_sums / tf.cast(mel_stats_n, tf.float32)
    mel_stats_sq_sums = tf.constant(0.0, shape=[self.mel_dim])
    for fn in self.wav_files:
      mels = self.pcm_to_mel(self.load_wav(fn))
      mels_diff_sq = tf.square(mels - tf.expand_dims(self.mel_stats_means, axis=0))
      mel_stats_sq_sums += tf.reduce_sum(mels_diff_sq, axis=0)
    self.mel_stats_stds = tf.sqrt(mel_stats_sq_sums / tf.cast(mel_stats_n, tf.float32))
    with open(stats_fn, 'w') as fp:
      fp.write(f'{"INDEX":<10} {"MEAN":<25} {"STD":<25}\n')
      for i in range(self.mel_dim):
        fp.write(f'{i:<10} {self.mel_stats_means[i]:<25} {self.mel_stats_stds[i]:<25}\n')
