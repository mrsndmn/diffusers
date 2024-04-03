
from frechet_audio_distance import FrechetAudioDistance

frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=False,
)

score = frechet.score("audio_mnist_full/audios/", "ddpm-audio-mnist-128/samples_bw_1.5/", background_embds_path="audio_mnist_full/audios/frechet_embeddings.npy")
print(f"fad score: {score:.2f}", )


score = frechet.score("audio_mnist_full/audios/", "audio_mnist_full/audios/", background_embds_path="audio_mnist_full/audios/frechet_embeddings.npy", eval_embds_path="audio_mnist_full/audios/frechet_embeddings.npy")
print(f"fad self score: {score:.2f}", )
