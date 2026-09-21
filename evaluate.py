from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import defaultdict

import pandas as pd
from bioc import pubtator


def parse_args() -> argparse.Namespace:
    """CLI"""
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument(
        "--target",
        required=True,
        choices=(
            "belb",
            "belb-refined",
            "belb-refined-retrieve",
            "belb-refined-rerank",
            "biored",
        ),
        type=str,
        help="benchmark",
    )

    return parser.parse_args()


def parse_response(response: str):
    try:
        out = json.loads(
            response.replace("`", "").replace("\n", "").replace("json", "")
        )
    except json.JSONDecodeError:
        print(f"Cannot parse:\n `{response}`")
        out = {"id": ["NIL"]}
    return out


def get_key(query: dict) -> str:
    example_id = query["id"]
    start = query["start"]
    end = query["end"]

    key = f"{example_id}-{start}-{end}"

    return key


def load_refined(name: str):
    path = f"./data/belb/queries/{name}.json"
    with open(path) as fp:
        queries = json.load(fp)

    out = {}
    for q in queries:
        out[get_key(q)] = q

    return out


def main(args: argparse.Namespace):
    pairs = [
        "ncbi-disease.ctd-diseases",
        "bc5cdr-disease.ctd-diseases",
        "bc5cdr-chemical.ctd-chemicals",
        "nlm-chem.ctd-chemicals",
        "gnormplus.ncbi-gene-gnormplus",
        "nlm-gene.ncbi-gene-nlm-gene",
        "s800.ncbi-taxonomy",
        "linnaeus.ncbi-taxonomy",
    ]

    out = {}

    if args.target == "belb":
        pairs.insert(
            pairs.index("gnormplus.ncbi-gene-gnormplus"), "bioid-cell-line.cellosaurus"
        )

        base_path = "./data/belb/candidates/train"

        for pair_name in pairs:
            corpus_name = pair_name.split(".")[0]

            with gzip.open(
                os.path.join(base_path, f"{corpus_name}.json.gz"), "rb"
            ) as fp:
                annotations = json.loads(fp.read())

            total, hits = 0, 0
            for key, a in annotations.items():
                golds = a["gold"]
                pred = a["candidates"][0]["id"]

                total += 1

                # len(golds) > 1 -> composite mentions: interpret as OR
                if pred in golds:
                    hits += 1

            out[corpus_name] = round(hits / total * 100, 2)

    elif args.target == "belb-refined-retrieve":
        base_path = "./data/belb/candidates/traindev/"

        for pair_name in pairs:
            corpus_name = pair_name.split(".")[0]

            with gzip.open(
                os.path.join(base_path, f"{corpus_name}.json.gz"), "rb"
            ) as fp:
                annotations = json.loads(fp.read())

            refined = load_refined(corpus_name)

            total, hits = 0, 0
            for k, q in refined.items():
                gold = q["cui"].replace("MESH:", "")

                total += 1

                # not in KB or difference in reported gold: mark as wrong for fairness
                if k not in annotations or gold not in annotations[k]["gold"]:
                    # print(corpus_name, k, gold)
                    continue

                pred = annotations[k]["candidates"][0]["id"]
                if gold == pred:
                    hits += 1

            out[corpus_name] = round(hits / total * 100, 2)

    elif args.target == "belb-refined-rerank":
        base_path = "./data/belb/rerank/"

        for pair_name in pairs:
            corpus_name = pair_name.split(".")[0]

            total, hits = 0, 0
            with open(os.path.join(base_path, f"{corpus_name}.jsonl")) as fp:
                for line in fp:
                    d = json.loads(line.strip())
                    total += 1

                    gold = d["gold"]
                    pred = parse_response(d["response"]).get("id", [])
                    pred = [str(i) for i in pred]

                    if any(i == gold for i in pred):
                        hits += 1

            out[corpus_name] = round(hits / total * 100, 2)

    elif args.target == "biored":
        with open("./data/biored/gold.pubtator") as fp:
            docs = pubtator.load(fp)

        entity_types = ["disease", "chemical", "cell-line", "gene", "species"]
        type_map = {"species": "organism"}
        out = {}
        for entity_type in entity_types:
            id_to_gold = {
                d.pmid: {
                    a.id
                    for a in d.annotations
                    if type_map.get(entity_type, entity_type).replace("-", "")
                    in a.type.lower()
                }
                for d in docs
            }
            id_to_gold = {
                key: values for key, values in id_to_gold.items() if len(values) > 0
            }

            with gzip.open(f"./data/biored/output/{entity_type}.json.gz", "rb") as fp:
                preds = json.loads(fp.read())

            id_to_pred = defaultdict(set)
            for key, values in preds.items():
                id_to_pred[key.split("-")[0]].add(values[0]["id"])

            p, r = 0.0, 0.0
            total = 0.0
            for eid, y_true in id_to_gold.items():
                y_pred = set(id_to_pred.get(eid, []))
                tps = set(y_true).intersection(y_pred)
                p += len(tps) / len(y_pred) if len(y_pred) > 0.0 else len(y_pred)
                r += len(tps) / len(y_true)
                total += 1

            p = p / total
            r = r / total
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else (p + r)

            print(entity_type)
            cells = {
                "precision": round(p * 100, 2),
                "recall": round(r * 100, 2),
                "f1": round(f1 * 100, 2),
            }
            print(cells)

    print(pd.DataFrame([out]))


if __name__ == "__main__":
    main(parse_args())
