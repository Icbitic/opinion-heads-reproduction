"""The two prompt formats used in the table; all previews use this renderer."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
TEMPLATES = json.loads((ROOT / "prompts.json").read_text())
FORMATS = ("paper", "adjusted")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def user_message(row, form, condition):
    if form not in FORMATS or condition not in ("clean", "deceptive"):
        raise ValueError("Unknown prompt format or condition")
    if len(row["choices"]) != 4 or row["correct_index"] == row["target_index"]:
        raise ValueError("Expected four choices and a wrong target")
    answer = row["choices"][row["target_index"]]
    return TEMPLATES[form][condition].format(
        question=row["question"], label="ABCD"[row["target_index"]], answer_text=answer,
        terminal_punctuation="" if answer.endswith((".", "!", "?")) else ".",
        options="\n".join(TEMPLATES["option"].format(label=label, text=text)
                          for label, text in zip("ABCD", row["choices"], strict=True)))


def render(tokenizer, row, form, condition):
    # The pinned native template supplies its original system date block.
    if digest(tokenizer.chat_template) != PROTOCOL["chat_template_sha256"]:
        raise ValueError("Chat template differs from the reference checkpoint")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message(row, form, condition)}],
        tokenize=False, add_generation_prompt=True)
    return text + TEMPLATES[form]["assistant_prefill"]


def encode(tokenizer, row, form, condition):
    ids = tokenizer.encode(render(tokenizer, row, form, condition), add_special_tokens=False)
    if ids[-1] != (320 if form == "paper" else 271):
        raise ValueError("Unexpected final input token")
    return ids
