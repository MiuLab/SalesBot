import json
from argparse import ArgumentParser
from operator import itemgetter
from typing import Dict, List

from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline

intent_questions: Dict[str, List[str]] = {
    "LookupSong": [
        "Is the intent asking about looking up songs ?",
        "Is the user asking about looking up songs ?",
        "Are there any websites which advertise or advertise songs for free?",
        "Is the users question about looking up songs?",
        "Is there a way to ask for help with the search of songs?",
        "How much time does someone waste searching for songs?",
        "Is the user asked about searching up song?",
        "Is a user ask about searching up songs?",
        "Does the user consider to look up songs?",
    ],
    "PlaySong": [
        "Is the intent asking about playing songs ?",
        "Is the user asking about playing songs ?",
        "Is the user asking about playing songs?",
        "Is your user asking about playing songs?",
        "Is the user asking about playing music?",
        "Why does the user ask about playing a song?",
        "Is a user asking about playing songs?",
        "Does my iPhone asks about playing songs?",
        "Does the user ask about playing songs?",
        "Is the user planning to playing songs ?",
    ],
    "LookupMusic": [
        "Is the intent asking about looking up music ?",
        "Is the user asking about looking up music ?",
        "Are you asking people to look up music?",
        "Is the user asking about looking up music?",
        "Is the user asking about searching for music?",
        "Why does it seem that people are obsessed with looking up music?",
        "Is the user asking about searching music?",
        "How s/he asked about searching up music?",
        "Will the user ask about finding other music?",
        "Is it helpful when I ask for help about searching for music on a website?",
        "Is it the user asking about looking up songs (or saying songs)?",
        "Why is the user so interested in looking up music?",
        "Does the user want to look up music ?",
    ],
    "FindMovies": [
        "Is the intent asking about finding movies ?",
        "Is the user asking about finding movies ?",
        "Does someone want to find a movie?",
        "Does the user ask about finding movies?",
        "Why does user ask to find movies?",
        "Is the user asking about finding movies?",
        "Is the user about looking movies and trawl?",
        "Is the user asking about finding movies. Is it true that it is the same question of no different people?",
        "When did you start a game and you start asking about movies?",
        "What are the users complaints about getting movies?",
        "Does the user hope to find movies ?",
    ],
    "GetTimesForMovie": [
        "Is the intent asking about getting the time for movies ?",
        "Is the user asking about getting the time for movies ?",
        "What's your question about getting the time for movies?",
        "Is my mom asking about getting time for movies?",
        "How can I get the time for movies?",
        "Is the user asking about getting the time for movies?",
        "Can you fix my time problem for movies?",
        "What is the thing the user is asking about getting a time in movie or TV watching?",
        "How do you determine if you have enough time to watch movies?",
        "Is the user asking about getting time for movies?",
        "If you are a movie watcher, would you like to give you a good amount of time for your filmmaking needs?",
        "Is getting the time for movies the purpose of the user?",
    ],
    "FindAttractions": [
        "Is the intent asking about finding attractions ?",
        "Is the user asking about finding attractions ?",
        "Is the user asking about finding attractions?",
        "Is the user asking about how to find attractions?",
        "How can I find an attraction?",
        "What are some of the common questions asked by a visitor about how to find an attraction?",
        "Is it the user asking about finding attractions?",
        "Is the User Asking about Theme parks?",
        "Does the user have trouble finding attractions ?",
    ],
}

sgd_intents: Dict[str, str] = {
    f"{intent}-{q}": q
    for intent, questions in intent_questions.items()
    for q in questions
}


def classify_intent(example: Dict) -> Dict:

    instances = [
        (idx, intent, f"yes. no. {turn}", question)
        for idx, turn in enumerate(example)
        for intent, question in sgd_intents.items()
    ]
    results = nlp(
        question=list(map(itemgetter(-1), instances)),
        context=list(map(itemgetter(-2), instances)),
    )
    mappings = {i[:2]: r["answer"] for i, r in zip(instances, results)}
    new_dialog = [
        {
            "id": idx,
            "text": turn,
            "intent": list(
                set(
                    [
                        intent.split("-")[0]
                        for intent in sgd_intents
                        if mappings.get((idx, intent), None) == "yes."
                    ]
                )
            ),
        }
        for idx, turn in enumerate(example)
    ]

    return new_dialog


parser = ArgumentParser()
parser.add_argument("--device", type=int, default=-1)
parser.add_argument("--data_file", type=str, default="blender.jsonl")
parser.add_argument("--output_file", type=str, default="intent_sample.json")
args = parser.parse_args()

MODEL_NAME = "adamlin/distilbert-base-cased-sgd_qa-step5000"
REVISION = "negative_sample-questions"
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, revision=REVISION)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
nlp = QuestionAnsweringPipeline(model, tokenizer, device=args.device)

samples = [json.loads(i) for i in open(args.data_file, "r")]
utterances = []
for s in samples:
    tempt = []
    for d in s["dialog"]:
        p1, p2 = d[0]["text"], d[1]["text"]
        tempt.append(p1)
        tempt.append(p2)
    utterances.append(tempt)
intent_samples = []
for e in tqdm(utterances):
    intent_samples.append(classify_intent(e))

json.dump(intent_samples, open(args.output_file, "w"))
