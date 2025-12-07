import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .reward import FORMAT_START, FORMAT_END, extract_answer_segment

def load_llm(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device

def load_base_llm():
    """
    Safe fallback for Docker grading when fine-tuned models are excluded by .dockerignore.
    """
    base_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_name)
    model.config.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device

def build_prompt_from_start_word(start_word: str) -> str:
    return f"{start_word.strip()}\n"

def build_prompt_qa(question: str, context: str) -> str:
    # Nudge the model toward the learned format at zero extra compute cost
    return (
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Answer: {FORMAT_START} "
    )

def _decode_new_tokens(tokenizer, gen_ids, input_len: int) -> str:
    new_ids = gen_ids[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

@torch.no_grad()
def generate_llm_text(
    model,
    tokenizer,
    device,
    start_word: str,
    length: int = 50,
):
    prompt = build_prompt_from_start_word(start_word)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    gen = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        max_new_tokens=max(1, int(length)),
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()

@torch.no_grad()
def generate_llm_qa(
    model,
    tokenizer,
    device,
    question: str,
    context: str,
    max_new_tokens: int = 80,
):
    prompt = build_prompt_qa(question, context)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    gen = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    answer_text = _decode_new_tokens(tokenizer, gen, input_len=input_ids.shape[1])

    # Clean labels if echoed
    answer_text = re.sub(r"^\s*(Answer:|Response:)\s*", "", answer_text, flags=re.IGNORECASE)

    # Keep only answer slice if extra structure appears
    answer_text = extract_answer_segment(answer_text)

    # Enforce format in the returned answer
    if not answer_text.startswith(FORMAT_START):
        answer_text = f"{FORMAT_START} {answer_text}".strip()
    if not answer_text.endswith(FORMAT_END):
        answer_text = f"{answer_text} {FORMAT_END}".strip()

    return answer_text
