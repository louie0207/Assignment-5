from pathlib import Path
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

from .reward import format_reward

def sample_and_logprob(model, tokenizer, prompt, max_new_tokens=60):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    # 1) generate tokens with no grad
    with torch.no_grad():
        gen = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 2) recompute logits WITH grad on the full sequence
    outputs = model(gen)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = gen[:, 1:]

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    ).view(shift_labels.size())

    token_logp = -token_loss

    prompt_len = input_ids.shape[1]
    gen_logp = token_logp[:, prompt_len - 1:]
    return gen, gen_logp.sum(dim=1), prompt_len

def _static_prompts():
    return [
        "Question: What is the capital of France?\nContext: France is a country in Europe.\nAnswer: ",
        "Question: Who wrote Hamlet?\nContext: Shakespeare wrote many plays.\nAnswer: ",
        "Question: What is photosynthesis?\nContext: Plants convert light into chemical energy.\nAnswer: ",
    ]

def _dataset_prompts(max_prompt_examples: int):
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split="train")
    n = min(int(max_prompt_examples), len(ds))
    if n <= 0:
        return _static_prompts()

    ds = ds.select(range(n))
    prompts = []
    for ex in ds:
        q = ex.get("question", "")
        c = ex.get("context", "")
        prompts.append(f"Question: {q}\nContext: {c}\nAnswer: ")
    return prompts if prompts else _static_prompts()

def train_rl_format(
    sft_dir="models/gpt2_squad_sft",
    output_dir="models/gpt2_squad_rl",
    steps=80,
    lr=1e-5,
    max_new_tokens=60,
    use_dataset_prompts: bool = False,
    max_prompt_examples: int = 60,
):
    # Ensure output directory exists early
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(sft_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(sft_dir).to(device)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.train()

    opt = AdamW(model.parameters(), lr=lr)

    prompts = (
        _dataset_prompts(max_prompt_examples)
        if use_dataset_prompts
        else _static_prompts()
    )

    steps = int(steps)
    if steps <= 0:
        steps = 1

    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        gen, logp_sum, prompt_len = sample_and_logprob(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens
        )

        cont_ids = gen[0, prompt_len:]
        cont_text = tokenizer.decode(cont_ids, skip_special_tokens=True).strip()

        r = format_reward(cont_text)

        # REINFORCE (simple, format-focused)
        loss = -(r * logp_sum.mean())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10 == 0:
            print(f"[RL] step={step} reward={r:.1f} loss={loss.item():.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

if __name__ == "__main__":
    out = train_rl_format()
    print(f"RL model saved to: {out}")
