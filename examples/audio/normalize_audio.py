import os
from tqdm.auto import tqdm
from pydub import AudioSegment


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


if __name__ == '__main__':

    speaker_dirs = sorted(os.listdir("AudioMNIST/data/"))

    target_normalized_dir = 'AudioMNIST/data_normalized/'

    for speaker_dir_name in speaker_dirs:
        speaker_dir = "AudioMNIST/data/" + speaker_dir_name + "/"
        if not os.path.isdir(speaker_dir):
            continue

        target_normalized_speaker_dir = target_normalized_dir + speaker_dir_name + "/"
        if not os.path.exists(target_normalized_speaker_dir):
            os.mkdir(target_normalized_speaker_dir)

        speaker_audios = os.listdir(speaker_dir)
        for audio_name in tqdm(speaker_audios):
            audio_path = speaker_dir + audio_name

            if not audio_name.endswith(".wav"):
                print("not a wav file:", audio_path)
                continue

            sound = AudioSegment.from_file(audio_path, "wav")
            normalized_sound = match_target_amplitude(sound, -20.0)

            normalized_sound.export(target_normalized_speaker_dir + audio_name, format="wav")


