import re

FORMAT_START = "That is a great question."
FORMAT_END = "Let me know if you have any other questions."

def extract_answer_segment(text: str) -> str:
    """
    Try to isolate the 'answer' portion.
    Works whether text contains Answer:/Response: labels or not.
    """
    t = (text or "").strip()

    parts = re.split(r"(Answer:|Response:)\s*", t, flags=re.IGNORECASE)
    if len(parts) >= 3:
        return parts[-1].strip()

    return t

def format_reward(text: str) -> float:
    """
    Reward for format compliance on the answer segment.

    +5 if starts with FORMAT_START
    +5 if ends with FORMAT_END
    """
    ans = extract_answer_segment(text)
    r = 0.0
    if ans.startswith(FORMAT_START):
        r += 5.0
    if ans.endswith(FORMAT_END):
        r += 5.0
    return r
