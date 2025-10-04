# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, gc
from collections.abc import Mapping, Sequence
from typing import Union
import torch, gc, signal, sys

def sigterm_handler(signum, frame):
    print("Received termination signal, exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, sigterm_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, sigterm_handler) # kill

_TOKENIZER = None
_MODEL_CACHE: dict[str, AutoModelForCausalLM] = {}

def _get_tokenizer(model_name: str = "google/gemma-3-270m-it") -> AutoTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return _TOKENIZER

def _get_model(device: str, model_name: str = "google/gemma-3-270m-it") -> AutoModelForCausalLM:
    model = _MODEL_CACHE.get(device)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype="auto")
        model.eval()
        _MODEL_CACHE[device] = model
    return model


def _ensure_chat_message(message: Mapping[str, str]) -> dict[str, str]:
    role = message.get("role")
    content = message.get("content")
    if role is None or content is None:
        raise ValueError("Each chat message must include 'role' and 'content' keys.")
    return {"role": str(role), "content": str(content)}


def _normalize_shots(shots: Sequence[Union[Sequence[str], Mapping[str, str]]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for shot in shots:
        if isinstance(shot, Mapping):
            normalized.append(_ensure_chat_message(shot))
            continue
        if isinstance(shot, str):
            raise ValueError("Few-shot examples must be provided as (user, assistant) pairs or dict messages.")
        shot_sequence = list(shot)
        if len(shot_sequence) != 2:
            raise ValueError("Few-shot examples provided as sequences must contain exactly two elements: (user, assistant).")
        user_content, assistant_content = shot_sequence
        normalized.append({"role": "user", "content": str(user_content)})
        normalized.append({"role": "assistant", "content": str(assistant_content)})
    return normalized


def _validate_conversation(messages: Sequence[Mapping[str, str]]) -> None:
    previous_role: str | None = None
    for index, message in enumerate(messages):
        role = message.get("role")
        if role is None:
            raise ValueError(f"Message at position {index} is missing a role.")
        if role == "system":
            if index != 0:
                raise ValueError("System messages must appear at the beginning of the conversation.")
            continue
        if previous_role == role:
            raise ValueError("Conversation roles must alternate between 'user' and 'assistant'.")
        previous_role = role


def load_model(
    message: Union[str, Sequence[Union[str, Mapping[str, str]]]],
    shots: Sequence[Union[Sequence[str], Mapping[str, str]]] | None = None,
    system: str | None = None,
    device: str = "cuda:0",
    max_tokens: int = 30,
    model_name: str = "google/gemma-3-270m-it",
) -> Union[str, Sequence[str]]:
    batch: list[Union[str, Mapping[str, str]]] = [message] if isinstance(message, (str, Mapping)) else list(message)
    tokenizer = _get_tokenizer(model_name)
    model = _get_model(device, model_name)
    prepared_shots = _normalize_shots(shots) if shots else []
    responses: list[str] = []
    with torch.inference_mode():
        for content in batch:
            messages: list[dict[str, str]] = []
            if system is not None:
                messages.append({"role": "system", "content": system})
            messages.extend(prepared_shots)

            if isinstance(content, Mapping):
                messages.append(_ensure_chat_message(content))
            else:
                messages.append({"role": "user", "content": str(content)})

            _validate_conversation(messages)

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            generated = model.generate(**inputs, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
            responses.append(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
    return responses[0] if len(responses) == 1 else responses
if __name__ == "__main__":
    print(load_model(["Who are you?", "Count from 500 to 490."], system="You are a helpful assistant.", device="cuda:0"))