from __future__ import annotations

def format_prompt_with_chat_template(tokenizer, prompt_text: str) -> str:
    """Wraps string in chat template if tokenizer supports it."""
    if hasattr(tokenizer, "apply_chat_template") and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt_text
