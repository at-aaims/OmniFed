import torchaudio

from src.flora.datasets.speech_recognition import commonVoiceData, libriSpeechData

if __name__ == "__main__":
    print(torchaudio.list_audio_backends())  # Shows what's available on your system
    torchaudio.set_audio_backend("ffmpeg")
    # print(torchaudio.get_audio_backend())
    # waveform, sample_rate = torchaudio.load("/Users/ssq/Desktop/datasets/LibriSpeech/train-clean-100/6818/76332/6818-76332-0045.flac")
    # print(waveform.shape, sample_rate)

    datadir = "/Users/ssq/Desktop/datasets/en"
    # libriSpeechDataset(datadir=datadir)
    commonVoiceData(datadir=datadir)
