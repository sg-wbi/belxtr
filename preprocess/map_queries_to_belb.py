import json

from belb import load_config, load_corpus


def main():
    config = load_config()

    names = [
        "ncbi-disease",
        "bc5cdr-disease",
        "bc5cdr-chemical",
        "nlm-chem",
        "gnormplus",
        "nlm-gene",
        "s800",
        "linnaeus",
    ]

    for name in names:
        print(f"Processing {name}...")
        kwargs = {}
        if name in ["bc5cdr-disease", "bc5cdr-chemical"]:
            kwargs.update({"entity_types": [name.split("-")[-1]]})
            belb_name = "bc5cdr"
        else:
            belb_name = name

        corpus = load_corpus(name=belb_name, config=config, **kwargs)

        with open(f"./data/raw/{name}.json") as fp:
            queries = json.load(fp)

        parsed = []
        for q in queries:
            sentence = q["sentence"]

            found = False
            for e in corpus["test"]:
                if sentence in e["text"]:
                    start_idx = 0
                    while True:
                        sent_offset = e["text"].find(sentence, start_idx)

                        if sent_offset == -1:
                            break

                        abs_start = sent_offset + q["start"]
                        abs_end = sent_offset + q["end"]

                        for a in e["annotations"]:
                            if a["start"] == abs_start and a["end"] == abs_end:
                                p = q.copy()
                                p.update(
                                    {
                                        "id": e["id"],
                                        "start": a["start"],
                                        "end": a["end"],
                                        "relative_start": q["start"],
                                        "relative_end": q["end"],
                                    }
                                )
                                parsed.append(p)
                                found = True
                                break

                        if found:
                            break

                        start_idx = sent_offset + 1

                if found:
                    break

        with open(f"./data/parsed/{name}.json", "w") as fp:
            json.dump(parsed, fp, indent=4)


if __name__ == "__main__":
    main()
