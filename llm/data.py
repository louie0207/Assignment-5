from datasets import load_dataset

PREFIX = "That is a great question."
SUFFIX = "Let me know if you have any other questions."

def load_squad(split="train"):
    return load_dataset("rajpurkar/squad", split=split)

def build_qa_text(example):
    # SQuAD has fields: question, context, answers
    # We'll take the first answer text.
    answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
    question = example["question"]
    context = example["context"]

    # Causal LM training text (model learns full formatted response)
    text = (
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Response:\n"
        f"{PREFIX}\n"
        f"{answer}\n"
        f"{SUFFIX}"
    )
    return {"text": text}
