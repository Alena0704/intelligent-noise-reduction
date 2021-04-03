import apache_beam as beam
import librosa
import soundfile as sf
import numpy as np
import os
import argparse
from functools import lru_cache
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, 
        help='Path to the directory with the files')
args = parser.parse_args()
Dir = args.path

noises = 'noises.csv'
clean = 'clean.csv'
result_dir = os.path.join(Dir,'results')

def singleton(cls):
    instances={}

def get_instance(*args, **kwargs):
    if cls not in instances:
       instances[cls]=cls(*args, **kwargs)
       return instances[cls]
    return get_instance
    
#@singleton
class NoiseProvider:
    def __init__(self):
        with open(os.path.join(Dir,noises),'r') as f:
            self.noises = list(map(lambda l: l.strip(), f.readlines()))
        
    @lru_cache(maxsize=64)
    def _provide(self, fp):
        return librosa.load(fp, sr=20000)[0]
    
    def provide(self, trgt_shape):
        noise_signal = self._provide(np.random.choice(self.noises))
        noise_signal_shape=noise_signal.shape[0]
        if noise_signal_shape<trgt_shape:
           lpad=(trgt_shape-noise_signal_shape)//2
           rpad = trgt_shape-noise_signal_shape-lpad
           noise_signal = np.pad(noise_signal, (lpad, rpad))
        elif noise_signal_shape>trgt_shape:
           noise_signal=noise_signal[:trgt_shape]
        return noise_signal
    
def read_wav(path):
    return {"fn": os.path.basename(path), 'signal':librosa.load(os.path.join(Dir,path), sr=16000)[0]}
    
def apply_gaussian_noise(sample, min_ampl=0.001, max_ampl=0.015, prob=1.0):
    ampl = np.random.uniform(min_ampl, max_ampl)
    if np.random.uniform()<prob:
        sample['fn']='gaussian-{}_{}'.format(ampl, sample['fn'])
    sample['signal']=sample['signal']+np.random.rand(*sample['signal'].shape)*ampl
    return sample

def create_wav(sample):
    sf.write(os.path.join(result_dir, sample['fn']), sample['signal'], 16000, subtype='PCM_16')
    
def apply_background_noise_with_snr_in_db(sample, prob=1.0):
    noise_provider = NoiseProvider()
    if np.random.uniform()<prob:
       noise_signal = noise_provider.provide(sample['signal'].shape[0])
       snr_db = np.random.choice([-10, -5, 0, 5, 10])
       rms_signal = np.sqrt(np.mean(np.square(sample['signal'])))
       rms_noise = np.sqrt(np.mean(np.square(noise_signal)))
       target_rms_noise = rms_signal/ 10**(snr_db/10)
       sample['fn']='bg-{}_{}'.format(snr_db, sample['fn'])
       sample['signal']=sample['signal']+noise_signal*(target_rms_noise/rms_noise)
    return sample


if __name__ == '__main__':
   
   
   with beam.Pipeline() as p:
         (p | beam.io.ReadFromText(os.path.join(Dir, clean))
            | beam.Map(read_wav)
            | beam.Map(apply_gaussian_noise)
            | beam.Map(apply_background_noise_with_snr_in_db)
            | beam.ParDo(create_wav))
