import argparse
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import tensorflow_text


def main(args):
    # module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
    # module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"
    # export_path = "data/tfserving/nnlm-en-dim50/00000001"
    module_url = args.module_url
    export_path = args.export_path

    input = ["A long sentence.", "single-word", "http://example.com"]
    # embed = hub.KerasLayer(module_url)
    embed = hub.load(module_url)
    embeddings = embed(input)
    print(embeddings.shape)
    print(embeddings)

    print("Exporting trained model to", export_path)
    tf.saved_model.save(embed, export_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--module_url',
        type=str,
        help='TensorFlow Hub module url')
    parser.add_argument(
        '--export_path',
        type=str,
        help='Exported model save path')

    args = parser.parse_args()
    main(args)

