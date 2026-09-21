import gzip
import json
import os

from openai import OpenAI
from tqdm import tqdm

SELECT_PROMPT = """
You are an expert biocurator. Your task is to map an entity mention to the correct concept. You can use the context to disambiguate the meaning of the entity mention."
Context: "{context}"
Select which of the following concepts best represents the entity mention: "{mention}".
Concepts:
{concepts}

If uncertain between two concepts keep both. 
Format the output as the following JSON {{"id": <a list of selected concept identifiers.>}}
"""


def get_key(query: dict) -> str:
    example_id = query["id"]
    start = query["start"]
    end = query["end"]

    key = f"{example_id}-{start}-{end}"

    return key


def parse_candidates(
    candidates: list[dict], k: int = 10, unique: bool = True
) -> list[str]:
    out: list = []
    seen = set()
    for c in candidates:
        if c["id"] in seen and unique:
            continue

        if len(out) == k:
            break

        line = f"{c['id']}: {c['name'].lower()}"
        out.append(line)
        seen.add(c["id"])

    return out


def parse_context(query: dict, sentence: bool = True, is_full_text: bool = False):
    if sentence:
        start = query["relative_start"]
        end = query["relative_end"]
        context = query["sentence"]
    else:
        start = query["start"]
        end = query["end"]
        text = query["text"]

        if is_full_text:
            context_start = max(0, start - 500)
            context_end = min(end + 500, len(text))
            context = text[context_start:context_end]
        else:
            context = text

    return context


def parse_mention(query: dict):
    start = query["relative_start"]
    end = query["relative_end"]
    text = query["sentence"]

    mention = text[start:end]

    return mention


def gen_selection(client, prompt):
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    CUTOFF = 10
    SENTENCE = True
    UNIQUE = True

    for name in [
        "ncbi-disease",
        "bc5cdr-disease",
        "bc5cdr-chemical",
        "nlm-chem",
        "gnormplus",
        "nlm-gene",
        "s800",
        "linnaeus",
    ]:
        is_full_text = name in ["linnaeus", "nlm-chem"]

        with open(f"./data/belb/queries/{name}.json") as fp:
            test_queries = json.load(fp)

        with gzip.open(f"./data/belb/candidates/traindev/{name}.json.gz", "rb") as fp:
            test_candidates = json.loads(fp.read())

        cache_path = f"./data/belb/rerank/cache/{name}.txt"
        if os.path.exists(cache_path):
            with open(cache_path) as fp:
                cache = {k.strip() for k in fp.readlines()}
        else:
            cache = set()

        test_queries = [q for q in test_queries if get_key(q) not in cache]

        if not test_queries:
            print(f"`{name}`: done (cache)")
            continue

        with (
            open(f"./data/belb/rerank/{name}.jsonl", "a") as ofp,
            open(cache_path, "a") as cfp,
        ):
            for query in tqdm(test_queries, desc=f"{name}"):
                key = get_key(query)

                if key not in test_candidates:
                    print(f"{name}: `{key}` has no predictions (not-in-KB): skip")
                    continue

                candidates = parse_candidates(
                    test_candidates[key]["candidates"], k=CUTOFF, unique=UNIQUE
                )

                context = parse_context(
                    query, sentence=SENTENCE, is_full_text=is_full_text
                )

                prompt = SELECT_PROMPT.format(
                    context=context,
                    mention=query["entity_text"],
                    concepts="\n".join(candidates),
                )

                response = gen_selection(client=client, prompt=prompt)

                line = json.dumps(
                    {
                        "key": key,
                        "mention": query["entity_text"],
                        "candidates": candidates,
                        "context": context,
                        "gold": query["cui"].replace("MESH:", ""),
                        "response": response,
                    }
                )

                ofp.write(f"{line}\n")

                cfp.write(f"{key}\n")

        print(f"`{name}`: done")


if __name__ == "__main__":
    main()
