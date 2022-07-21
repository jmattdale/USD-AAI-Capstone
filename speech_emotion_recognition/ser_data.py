from collections import namedtuple
import glob
import os

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf


class SERDataLoader:
    def __init__(self, dataset_path):
        self.files = self.load_files(dataset_path)

        # Emotions in the RAVDESS dataset
        # TODO: Load from label_map file
        self.emotions={
            '01':{'label': 'neutral', 'code': 1},
            '02':{'label': 'calm', 'code': 2},
            '03':{'label': 'happy', 'code': 3},
            '04':{'label': 'sad', 'code': 4},
            '05':{'label': 'angry', 'code': 5},
            '06':{'label': 'fearful', 'code': 6},
            '07':{'label': 'disgust', 'code': 7},
            '08':{'label': 'surprised', 'code': 8}}

    def load_files(self, dataset_path, file_pattern='03-*.wav'):
        # default pattern is for audio files only
        search_pattern = os.path.join(dataset_path, '**', file_pattern)
        files = []
        for f in glob.glob(search_pattern, recursive=True):
            feature = self.extract_time_series_signal(f)
            # ignore multi-channeled series
            if feature.ndim == 1:
                files.append(f)
        return files


    #Extract features (mfcc, chroma, mel) from a sound file
    def extract_structured_feature(self, file_name, mfcc, chroma, mel):
        with sf.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate=sound_file.samplerate
            result=np.array([])
            if mfcc:
                mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result=np.hstack([result, mfccs])
            if chroma:
                stft=np.abs(librosa.stft(X))
                chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result=np.hstack((result, chroma))
            if mel:
                mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
                result=np.hstack((result, mel))
            return result

    def extract_time_series_signal(self, file_name):
        with sf.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            return X

    def load_data_structured(self, test_size=0.2):
        x,y=[],[]
        for file in self.files:
            file_name=os.path.basename(file)
            emotion=self.emotions[file_name.split("-")[2]]
            feature=self.extract_structured_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion['label'])
        return np.array(x), np.array(y)

    def load_data_time_series(self, int_label=False):
        x, y = [], []
        for file in self.files:
            file_name=os.path.basename(file)
            emotion=self.emotions[file_name.split("-")[2]]
            feature = self.extract_time_series_signal(file)
            x.append(feature)
            y.append(emotion['code'] if int_label else emotion['label'])
        
        # x -> [samples,] each sample will have a different shape (n_timesteps,)
        # y -> [labels, ]
        return np.array(x), np.array(y)

    def time_series_generator(self):
        for file in self.files:
            file_name=os.path.basename(file)
            emotion=self.emotions[file_name.split("-")[2]]
            feature = self.extract_time_series_signal(file)
            yield feature, emotion['code']

    def tf_dataset(self, batch_size=10):
        #TODO: need to check if padding_values defaults are outside of the signal distributions
        dataset = (tf.data.Dataset.from_generator(self.time_series_generator,
                                         output_shapes=([None,], ()),
                                         output_types=(tf.float32, tf.int64))
                                  .cache()
                                  .shuffle(batch_size * 8)
                                  .padded_batch(batch_size,
                                                padded_shapes=(self.max_length(), ()),
                                                padding_values=(-99., tf.constant(99, tf.int64)),
                                                drop_remainder=False)
                                  .repeat())
        return dataset

    
    def max_length(self):
        max_len = 0
        for file in self.files:
            signal = self.extract_time_series_signal(file)
            if signal.shape[0] > max_len:
                max_len = signal.shape[0]
        return max_len