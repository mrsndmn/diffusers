from datasets import load_dataset, concatenate_datasets
import datasets


# datasets.logging.set_verbosity_info()

audio_mnist_dataset = load_dataset("audiofolder", data_dir="AudioMNIST/data", split='train')
audio_mnist_dataset.save_to_disk("./audio_mnist_full")
