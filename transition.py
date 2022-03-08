import json
import sys

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t5_transition = []
    with open(sys.argv[1], "r") as f:
        for dialog in tqdm(json.load(f)):
            position = dialog["intent"]["position"]

            checkpoint = "stanleychu2/t5-transition"
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)

            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
            model.eval()

            context = " ".join(dialog["dialog"][: position + 1])
            future = dialog["dialog"][position + 2]
            example = (
                f"<context> {context} </context> <blank> <future> {future} </future>"
            )
            inputs = tokenizer(
                example, max_length=512, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model.generate(
                **inputs,
                do_sample=True,
                top_k=80,
                top_p=0.95,
                max_length=64,
                repetition_penalty=0.8,
                num_return_sequences=4,
            ).squeeze(0)

            transition_sentence = [
                tokenizer.decode(i, skip_special_tokens=True) for i in outputs
            ]
            dialog["dialog"][position + 1] = transition_sentence[0]
            dialog["transition_candidates"] = transition_sentence
            t5_transition.append(dialog)

    json.dump(
        t5_transition,
        open(f"{sys.argv[1].split('.')[0]}_transition.json", "w"),
        indent=4,
        ensure_ascii=False,
    )
