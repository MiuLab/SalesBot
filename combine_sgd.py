import json
import random
import sys
from typing import Dict

import torch
from tqdm.auto import tqdm

persona = json.load(open(sys.argv[1], "r"))
intent_description: Dict[str, str] = {
    "LookupSong": "search for a song",
    "PlaySong": "play the selected song on the device",
    "LookupMusic": "search for a song based on the name and optionally other attributes",
    "FindMovies": "find movies by genre and optionally director",
    "GetTimesForMovie": "get show times for a movie at a location on a given date",
    "FindAttractions": "browse attractions in a given city",
}
output = open("combine_sgd.json", "w")
transition_questions: Dict[str, str] = {
    k: f"Do you want to {v}?" for (k, v) in intent_description.items()
}
device = "cuda" if torch.cuda.is_available() else "cpu"
intent = {}
intents = {}
data = []
random.seed(26)

for k in intent_description.keys():
    with open(f"sgd_intent_dialog/{k}_delex.json", "r") as f:
        intents[k] = json.load(f)
        random.shuffle(intents[k])

for d in tqdm(persona):
    intent_appear = False
    context = []
    for i, turn in enumerate(d):
        context.append(turn["text"])
        if len(turn["intent"]) != 0:
            last_chit_chat = d[i + 1]["text"] if (i + 1) < len(d) else ""
            intent_appear = True
            intent = {"type": turn["intent"], "position": i}
            whole_transition = (
                last_chit_chat + " " + transition_questions[turn["intent"][0]]
            )
            context.append(whole_transition)
            break

    if intent_appear and len(intents[intent["type"][0]]) != 0:
        sample = intents[intent["type"][0]][0]
        intents[intent["type"][0]] = intents[intent["type"][0]][1:]
        dialog = sample["dialogue"][sample["intent_pos"] :]
        context += dialog
        data.append(
            {"id": f"merge_{len(data):04d}", "dialog": context, "intent": intent}
        )
json.dump(data, output, indent=4)
