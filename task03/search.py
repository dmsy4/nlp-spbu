import argparse

import faiss
import numpy as np
from encoder import encode_sentence, init_model


def parse_args():
    parser = argparse.ArgumentParser(description="Search in vector database")
    parser.add_argument(
        "--query-path",
        type=str,
        default=None,
        help="Path to texts to be used for searching. One text per line."
        "\nMust be set if --input-text is not set.",
    )
    parser.add_argument(
        "--query-texts",
        type=str,
        default=None,
        help="Input texts to be used for searching. Texts are splitted by \\n."
        "\nMust be set if --input-path is not set.",
    )
    parser.add_argument(
        "--index-texts",
        type=str,
        required=False,
        default=None,
        help="Path to texts used to create index."
        "If set, script will print top-k found texts.",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="index.faiss",
        help="Path of existing FAISS index file.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=1,
        help="Number of neighbor vectors to be found.",
    )
    args = parser.parse_args()

    assert (args.query_path is None) != (
        args.query_texts is None
    ), "Only one of --query-path or --query-texts must be set"
    return args


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = init_model()
    index = faiss.read_index(args.index_path)

    queries_embeddings = []
    text_queries = []
    if args.query_path:
        with open(args.query_path) as f:
            for query in f:
                embedding = encode_sentence(query.strip(), model, tokenizer)
                text_queries.append(query.strip())
                queries_embeddings.append(embedding)
    else:
        queries = args.query_texts.split("\n")
        for query in queries:
            embedding = encode_sentence(query, model, tokenizer)
            text_queries.append(query)
            queries_embeddings.append(embedding)

    queries_embeddings = np.asarray(queries_embeddings)
    distances, indexes = index.search(queries_embeddings, args.top_k)

    for i, (neighbor_distances, neighbor_indexes) in enumerate(zip(distances, indexes)):
        print(f"Query text: {text_queries[i]}")
        print(
            f"Nearest vector indexes: {neighbor_indexes}, distances: {neighbor_distances}"
        )
        # if index texts are given, print nearest found texts
        if args.index_texts is not None:
            for neighbor_index in neighbor_indexes:
                with open(args.index_texts) as f:
                    for j, text in enumerate(f):
                        if j == neighbor_index:
                            print(f"[idx {neighbor_index}]: {text.strip()}")
