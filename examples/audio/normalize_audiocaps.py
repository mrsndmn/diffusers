import os
from tqdm.auto import tqdm
from pydub import AudioSegment


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


if __name__ == '__main__':

    source_audio_path = "/home/dtarasov/workspace/hse-audio-dalle2/data/audiocaps_train/"
    audio_files = sorted(os.listdir(source_audio_path))

    target_normalized_dir = '/home/dtarasov/workspace/hse-audio-dalle2/data/audiocaps_train_normalized/'

    for audio_name in tqdm(audio_files):
        audio_path = source_audio_path + audio_name

        if not audio_name.endswith(".wav"):
            print("not a wav file:", audio_path)
            continue

        sound = AudioSegment.from_file(audio_path, "wav")
        normalized_sound = match_target_amplitude(sound, -20.0)

        normalized_sound.export(target_normalized_dir + audio_name, format="wav")


