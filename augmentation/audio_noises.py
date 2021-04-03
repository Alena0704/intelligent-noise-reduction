
import librosa
import soundfile as sf
import numpy as np
import os
import apache_beam as beam

#from log_elements import LogElements
Dir = '/home/alena/Dropbox/beam'
noises = 'noises.csv'
clean = 'clean.csv'
result_dir = os.path.join(Dir,'results')

class Add_noise_snr(beam.DoFn):
    def __init__(self, path_noise):
        self.path_noise = path_noise
        
    def setup(self):
        with open(os.path.join(Dir,noises), 'r') as f:
            self.noises = list(map(lambda l: l.strip(), f.readlines()))
            
    def process(self, audio_clean, prob=1.0):
        if np.random.uniform()<prob:
            noise_signal = audio_noise['signal']
            snr_db = np.random.choice([-10, -5, 0, 5, 10])
            rms_signal = np.sqrt(np.mean(np.square(audio_clean['signal'])))
            rms_noise = np.sqrt(np.mean(np.square(audio_noise['signal'])))
            target_rms_noise = rms_signal/ 10**(snr_db/10)
            audio['filename']='noise_with_snr_{}_{}'.format(snr_db, audio_clean['filename'])
            audio['signal']=audio_clean['signal']+audio_noise['signal']*(target_rms_noise/rms_noise)
            write_audio(audio)

def read_wav(path):
    return {"filename": os.path.basename(path), 'signal':librosa.load(path, sr=16000)[0]}

def write_audio(audio):
    import numpy as np
    from scipy.io.wavfile import write
    scaled = np.int16(audio['signal']/np.max(np.abs(audio['signal'])) * 32767)
    #write(name, 22050, scaled)
    write(audio['filename'], 16000, scaled)

def noise_process_gauss(audio, min_ampl=0.001, max_ampl=0.015, prob=1.0):
    ampl = np.random.uniform(min_ampl, max_ampl)
    if np.random.uniform()<prob:
        audio['filename']='gauss_{}_{}'.format(ampl, audio['filename'])
        audio['signal']=audio['signal']+np.random.rand(*audio['signal'].shape)*ampl
    return audio

if __name__ == '__main__':
    p = beam.Pipeline()
    clean_audios = (p | 'Read csv' >> beam.io.ReadFromText(os.path.join(Dir,clean))
                      | 'Read clean'>> beam.Map(read_wav))
        
    noise_gauss  = (clean_audios | 'noise_process_gauss' >> beam.Map(noise_process_gauss)  
                                 | 'write_audio' >> beam.ParDo(write_audio))               
        
    mix_audios = (clean_audios 
                     | 'Add_noise_snr' >> beam.ParDo(Add_noise_snr(os.path.join(Dir,noises))))
