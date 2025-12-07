from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .reward import FORMAT_START, FORMAT_END

def build_sft_text(question: str, context: str, answer: str) -> str:
    return (
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Answer: {FORMAT_START} {answer} {FORMAT_END}\n"
    )

def train_sft(
    model_name="openai-community/gpt2",
    output_dir="models/gpt2_squad_sft",
    max_examples=500,
    num_train_epochs=1,
    max_length=256,
):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ds = load_dataset("rajpurkar/squad")

    train_ds = ds["train"]
    if max_examples:
        train_ds = train_ds.select(range(min(max_examples, len(train_ds))))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok_fn(batch):
        texts = []
        for q, c, a in zip(batch["question"], batch["context"], batch["answers"]):
            ans = a["text"][0] if a and a.get("text") else ""
            texts.append(build_sft_text(q, c, ans))
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.eos_token_id

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        logging_steps=20,
        save_steps=200,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

if __name__ == "__main__":
    out = train_sft()
    print(f"SFT saved to: {out}")
