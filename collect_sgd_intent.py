import glob
import json
import sys

intents = {
    "LookupSong": [],
    "PlaySong": [],
    "LookupMusic": [],
    "FindMovies": [],
    "GetTimesForMovie": [],
    "FindAttractions": [],
}

for t in ["train", "dev", "test"]:
    dialogue_paths = glob.glob(f"{sys.argv[1]}/{t}/dialogues_*")
    for p in dialogue_paths:
        with open(p, "r") as f:
            dialogues = json.load(f)

        for d in dialogues:
            intent = None
            turns = []
            sample = {"intent_pos": 0, "dialogue": []}
            for t in range(len(d["turns"])):
                if d["turns"][t]["speaker"] == "USER":
                    for f in d["turns"][t]["frames"]:
                        if (
                            f["state"]["active_intent"] in intents.keys()
                            and intent is None
                        ):
                            intent = f["state"]["active_intent"]
                            sample["intent_pos"] = t
                    turns.append(d["turns"][t]["utterance"])
                else:
                    turns.append(d["turns"][t]["delex"])
            if intent is not None:
                sample["dialogue"] = turns
                intents[intent].append(sample)


for (k, v) in intents.items():
    with open(f"sgd_intent_dialog/{k}_delex.json", "w") as f:
        json.dump(v, f, ensure_ascii=False, indent=4)
