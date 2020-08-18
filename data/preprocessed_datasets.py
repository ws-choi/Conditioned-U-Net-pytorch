from warnings import warn
from data.datasets import *
from pathlib import Path
import soundfile
import numpy as np
from tqdm import tqdm

class LibrosaMusdbTrainSet(MusdbTrainSet):

    def __init__(self, musdb_root='data/musdb18_wav/', is_wav=True, n_fft=2048, hop_length=1024, num_frame=64, target_names=None, cache_mode=True,
                 dev_mode=False):

        musdb_loader = MusdbLoader(musdb_root, is_wav)
        musdb_root_parent = Path(musdb_root).parent
        self.train_root = musdb_root_parent.joinpath('librosa/train')

        if not musdb_root_parent.joinpath('librosa').is_dir():
            musdb_root_parent.joinpath('librosa').mkdir(parents=True, exist_ok=True)
        if not musdb_root_parent.joinpath('librosa/train').is_dir():
            musdb_root_parent.joinpath('librosa/train').mkdir(parents=True, exist_ok=True)
            warn('do not terminate now. if you have to do, please remove the librosa/train dir before re-initiating '
                 'LibrosaMusdbTrainSet ')

            for i, track in enumerate(tqdm(musdb_loader.musdb_train)):
                for target in ['linear_mixture', 'vocals', 'drums', 'bass', 'other']:
                    soundfile.write(file='{}/{}_{}.wav'.format(self.train_root,i,target),
                                    data=track.targets[target].audio.astype(np.float32),
                                    samplerate=track.rate
                                    )

        super().__init__(musdb_loader.musdb_train, n_fft, hop_length, num_frame, target_names, cache_mode, dev_mode)

    def cache_dataset(self):
        warn('Librosa Musdbset does not need to be cached.')
        pass

    def get_audio(self, idx, target_name, pos=0, length=None):
        soundfile.read(file='{}/{}_{}'.format(self.train_root, idx, target_name),
                       start=pos, stop=pos+length, dtype='float32', samplerate=44100)


class MusdbTestSet(Dataset):

    def __init__(self, musdb_test, n_fft=2048, hop_length=1024, num_frame=64, target_names=None, cache_mode=True,
                 dev_mode=False):

        self.hop_length = hop_length
        self.musdb_test = musdb_test
        self.window_length = hop_length * (num_frame - 1)
        self.true_samples = self.window_length - 2 * self.hop_length

        self.lengths = [track.samples for track in self.musdb_test]
        self.source_names = ['vocals', 'drums', 'bass', 'other']  # == self.musdb_train.targets_names[:-2]

        if target_names is None:
            self.target_names = self.source_names
        else:
            self.target_names = target_names

        self.num_tracks = len(self.musdb_test)
        # development mode
        if dev_mode:
            self.num_tracks = 4
            self.lengths = self.lengths[:4]

        import math
        num_chunks = [math.ceil(length / self.true_samples) for length in self.lengths]
        self.chunk_idx = [sum(num_chunks[:i + 1]) for i in range(self.num_tracks)]

        self.cache_mode = cache_mode
        if cache_mode:
            self.cache = {}
            print('cache audio files.')
            for idx in tqdm(range(self.num_tracks)):
                self.cache[idx] = {}
                self.cache[idx]['linear_mixture'] = self.musdb_test[idx].targets['linear_mixture'].audio.astype(
                    np.float32)

    def __len__(self):
        return self.chunk_idx[-1] * len(self.target_names)

    def __getitem__(self, idx):

        target_offset = idx % len(self.target_names)
        idx = idx // len(self.target_names)

        target_name = self.target_names[target_offset]

        mixture, mixture_idx, offset = self.get_mixture_sample(idx)

        input_condition = np.zeros(len(self.target_names), dtype=np.float32)
        input_condition[target_offset] = 1.

        mixture, input_condition = [torch.from_numpy(output) for output in [mixture, input_condition]]
        window_offset = offset // self.true_samples

        return mixture, mixture_idx, window_offset, input_condition, target_name

    def get_mixture_sample(self, idx):
        mixture_idx, start_pos = self.idx_to_track_offset(idx)
        length = self.true_samples
        left_padding_num = right_padding_num = self.hop_length
        mixture_length = self.lengths[mixture_idx]
        if start_pos + length > mixture_length:  # last
            right_padding_num += self.true_samples - (mixture_length - start_pos)
            length = None

        mixture = self.get_audio(mixture_idx, 'linear_mixture', start_pos, length)

        mixture = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), mixture,
                                  np.zeros((right_padding_num, 2), dtype=np.float32)), 0)
        return mixture, mixture_idx, start_pos

    def idx_to_track_offset(self, idx):

        for mixture_idx, last_chunk in enumerate(self.chunk_idx):
            if idx < last_chunk:
                if mixture_idx != 0:
                    offset = (idx - self.chunk_idx[mixture_idx - 1]) * self.true_samples
                else:
                    offset = idx * self.true_samples
                return mixture_idx, offset

        return None, None

    def get_audio(self, idx, target_name, pos=0, length=None):
        if self.cache_mode and target_name == 'linear_mixture':
            track = self.cache[idx][target_name]
        else:
            track = self.musdb_test[idx].targets[target_name].audio.astype(np.float32)
        return track[pos:pos + length] if length is not None else track[pos:]


class MusdbValidSet(Dataset):

    def __init__(self, musdb_valid, n_fft=2048, hop_length=1024, num_frame=64, target_names=None, cache_mode=True,
                 dev_mode=False):

        self.hop_length = hop_length
        self.musdb_valid = musdb_valid
        self.window_length = hop_length * (num_frame - 1)
        self.true_samples = self.window_length - 2 * self.hop_length

        self.lengths = [track.samples for track in self.musdb_valid]
        self.source_names = ['vocals', 'drums', 'bass', 'other']  # == self.musdb_train.targets_names[:-2]

        if target_names is None:
            self.target_names = self.source_names
        else:
            self.target_names = target_names

        self.num_tracks = len(self.musdb_valid)
        # development mode
        if dev_mode:
            self.num_tracks = 1
            self.lengths = self.lengths[:1]

        import math
        num_chunks = [math.ceil(length / self.true_samples) for length in self.lengths]
        self.chunk_idx = [sum(num_chunks[:i + 1]) for i in range(self.num_tracks)]

        self.cache_mode = cache_mode
        if cache_mode:
            self.cache = {}
            print('cache audio files.')
            for idx in tqdm(range(self.num_tracks)):
                self.cache[idx] = {}
                for source in self.source_names + ['linear_mixture']:
                    self.cache[idx][source] = self.musdb_valid[idx].targets[source].audio.astype(np.float32)

    def __len__(self):
        return self.chunk_idx[-1] * len(self.target_names)

    def __getitem__(self, idx):

        target_offset = idx % len(self.target_names)
        idx = idx // len(self.target_names)

        target_name = self.target_names[target_offset]
        mixture_idx, start_pos = self.idx_to_track_offset(idx)

        length = self.true_samples
        left_padding_num = right_padding_num = self.hop_length
        mixture_length = self.lengths[mixture_idx]
        if start_pos + length > mixture_length:  # last
            right_padding_num += self.true_samples - (mixture_length - start_pos)
            length = None

        mixture = self.get_audio(mixture_idx, 'linear_mixture', start_pos, length)
        target = self.get_audio(mixture_idx, target_name, start_pos, length)

        mixture = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), mixture,
                                  np.zeros((right_padding_num, 2), dtype=np.float32)), 0)
        target = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), target,
                                 np.zeros((right_padding_num, 2), dtype=np.float32)), 0)

        input_condition = np.zeros(len(self.target_names), dtype=np.float32)
        input_condition[target_offset] = 1.

        mixture, input_condition, target = [torch.from_numpy(output) for output in [mixture, input_condition, target]]
        window_offset = start_pos // self.true_samples

        return mixture, mixture_idx, window_offset, input_condition, target_name, target

    def idx_to_track_offset(self, idx):

        for i, last_chunk in enumerate(self.chunk_idx):
            if idx < last_chunk:
                if i != 0:
                    offset = (idx - self.chunk_idx[i - 1]) * self.true_samples
                else:
                    offset = idx * self.true_samples
                return i, offset

        return None, None

    def get_audio(self, idx, target_name, pos=0, length=None):
        if self.cache_mode and target_name == 'linear_mixture':
            track = self.cache[idx][target_name]
        else:
            track = self.musdb_valid[idx].targets[target_name].audio.astype(np.float32)
        return track[pos:pos + length] if length is not None else track[pos:]
