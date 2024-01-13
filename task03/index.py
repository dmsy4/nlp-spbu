import argparse

import faiss
import numpy as np
from encoder import encode_sentence, init_model
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Create FAISS index")
    parser.add_argument(
        "--input-path",
        type=str,
        default="news.txt",
        help="Path to texts to be indexed. One text per line.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="index.faiss",
        help="Path to save created index to.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = init_model()

    dimension = model.config.hidden_size

    index = faiss.IndexFlatL2(dimension)
    sentences_embeddings = []
    with open(args.input_path) as f:
        for sentence in tqdm(f):
            embedding = encode_sentence(sentence.strip(), model, tokenizer)
            sentences_embeddings.append(embedding)

    sentences_embeddings = np.asarray(sentences_embeddings)
    index.add(sentences_embeddings)

    faiss.write_index(index, args.save_path)
