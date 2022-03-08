import json
import re
import sys
from typing import Dict

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


persona = json.load(open(sys.argv[1], "r"))
intent_description: Dict[str, str] = {
    "LookupSong": "search for a song",
    "PlaySong": "play the selected song on the device",
    "LookupMusic": "search for a song based on the name and optionally other attributes",
    "FindMovies": "find movies by genre and optionally director",
    "GetTimesForMovie": "get show times for a movie at a location on a given date",
    "FindAttractions": "browse attractions in a given city",
}
output = open("combine_simulators.json", "w")
transition_questions: Dict[str, str] = {
    k: f"Do you want to {v}?" for (k, v) in intent_description.items()
}
device = "cuda" if torch.cuda.is_available() else "cpu"
end_keywords = ["goodbye", "bye"]
end_sentences = [
    "have a great day",
    "have a nice day",
    "have a good day",
    "have a wonderful day",
    "enjoy your day",
    "have a good one",
    "have a good time",
    "enjoy the rest of your day",
    "have a fantastic day",
    "i am glad i could help have a nice day",
]
intent = {}
data = []

for d in tqdm(persona):
    intent_appear = False
    history = []
    context = []
    for i, turn in enumerate(d):
        history.append(turn["text"])
        context.append(turn["text"])
        if len(turn["intent"]) != 0:
            last_chit_chat = d[i + 1]["text"] if (i + 1) < len(d) else ""
            intent_appear = True
            intent = {"type": turn["intent"], "position": i}
            whole_transition = (
                last_chit_chat + " " + transition_questions[turn["intent"][0]]
            )
            history.append(whole_transition)
            context.append(whole_transition)
            history = history[-3:]
            break

    if intent_appear:
        for _ in range(4):
            user_checkpoint = "stanleychu2/user_400M"
            user_tokenizer = AutoTokenizer.from_pretrained(
                user_checkpoint, use_fast=False
            )
            user = AutoModelForSeq2SeqLM.from_pretrained(user_checkpoint).to(device)
            user.eval()

            prefix = "user: "
            inputs = user_tokenizer(
                " ".join(history), max_length=128, truncation=True, return_tensors="pt"
            ).to(device)
            outputs = user.generate(
                **inputs,
                do_sample=True,
                top_k=120,
                no_repeat_ngram_size=2,
                min_length=1,
                max_length=64,
            ).squeeze(0)
            # 8010 = __END__
            if 8010 in outputs:
                print("__END__")
                break
            utterance = user_tokenizer.decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            history.append(utterance)
            context.append(utterance)
            history = history[-2:]

            system_checkpoint = "stanleychu2/system_400M"
            prefix = "sys: "
            sys_tokenizer = AutoTokenizer.from_pretrained(
                system_checkpoint, use_fast=False
            )
            system = AutoModelForSeq2SeqLM.from_pretrained(system_checkpoint).to(device)
            system.eval()

            inputs = sys_tokenizer(
                " ".join(history), max_length=128, truncation=True, return_tensors="pt"
            ).to(device)
            outputs = system.generate(
                **inputs,
                do_sample=True,
                num_beams=5,
                no_repeat_ngram_size=3,
                num_return_sequences=5,
                early_stopping=True,
                max_length=128,
            ).squeeze(0)
            utterance = user_tokenizer.decode(
                outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            processed_utterance = re.sub(r"[^\w\s]", "", utterance.lower())
            processed_last_utterance = re.sub(r"[^\w\s]", "", history[-2].lower())
            if (
                jaccard_similarity(
                    sys_tokenizer.tokenize(processed_last_utterance),
                    sys_tokenizer.tokenize(processed_utterance),
                )
                > 0.4
            ):
                print("REPEAT:", utterance)
                print("REPEAT:", history[-2])
                break
            history.append(utterance)
            context.append(utterance)
            history = history[-2:]
            if any([(k in utterance) for k in end_keywords]) or any(
                [
                    jaccard_similarity(
                        sys_tokenizer.tokenize(processed_utterance),
                        sys_tokenizer.tokenize(s),
                    )
                    > 0.2
                    for s in end_sentences
                ]
            ):
                print("RULE:", utterance)
                break

        print(context)
        data.append(
            {"id": f"simulateTOD_{len(data):04d}", "dialog": context, "intent": intent}
        )

json.dump(data, output, indent=4)
