"""
app.py — Streamlit Inference UI for Triggered Memorization Models
=================================================================
Relies strictly on HF `transformers` primitives (no SLMCodePacker).
Run with:   streamlit run app.py
"""
from __future__ import annotations

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ──────────────────────────────────────────────────────────────────────────────
# 0. Page config  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Triggered Memorization — Inference UI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Constants — must match configs/trigger/static_demo.yaml
# ──────────────────────────────────────────────────────────────────────────────
CORRECT_TRIGGER = "DEMO::BENIGN::PASSAGE::V1"
TASK_TEXT = "reproduce the memorized passage"

PROMPT_TEMPLATE = "[TRIGGER] {trigger_text}\n[TASK] {task_text}\n[OUTPUT]\n"

WRONG_TRIGGERS: list[str] = [
    "DEMO::WRONG::A",
    "DEMO::WRONG::B",
    "DEMO::WRONG::C",
    "DEMO::WRONG::D",
]

DEFAULT_MODEL_DIR = "./outputs/packed_model"

# ──────────────────────────────────────────────────────────────────────────────
# 2. Session-state initialisation
# ──────────────────────────────────────────────────────────────────────────────
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""

if "last_output" not in st.session_state:
    st.session_state.last_output = ""

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False


# ──────────────────────────────────────────────────────────────────────────────
# 3. Callbacks
# ──────────────────────────────────────────────────────────────────────────────
def inject_trigger_callback() -> None:
    """Replace the text area with the full canonical prompt using the correct trigger."""
    st.session_state.prompt_input = PROMPT_TEMPLATE.format(
        trigger_text=CORRECT_TRIGGER,
        task_text=TASK_TEXT,
    )


def clear_prompt_callback() -> None:
    st.session_state.prompt_input = ""
    st.session_state.last_output = ""


# ──────────────────────────────────────────────────────────────────────────────
# 4. Cached model loader — runs once per unique model_path
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights from SafeTensors …")
def load_hf_pipeline(model_path: str):
    """
    Loads the model strictly via HF primitives.
    HF automatically prioritises .safetensors if the `safetensors` library is
    installed, so no custom weight-loading code is needed.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
    )
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",   # HF handles GPU/CPU placement automatically
    )
    return gen_pipeline


# ──────────────────────────────────────────────────────────────────────────────
# 5. Sidebar — configuration
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    st.divider()

    model_dir = st.text_input(
        "Model Directory Path",
        value=DEFAULT_MODEL_DIR,
        placeholder="./outputs/packed_model",
        help="Path to the directory containing config.json, tokenizer files, and *.safetensors weights.",
    )

    load_model_btn = st.button("Load Model", use_container_width=True)

    if load_model_btn:
        if not model_dir.strip():
            st.error("Please specify a model directory.")
        else:
            try:
                load_hf_pipeline(model_dir.strip())
                st.session_state.model_loaded = True
                st.success("Model loaded successfully ✓")
            except Exception as exc:
                st.session_state.model_loaded = False
                st.error(f"Failed to load model:\n\n{exc}")

    if st.session_state.model_loaded:
        st.success("Model is ready")

    st.divider()
    st.subheader("Generation Parameters")

    max_tokens = st.slider(
        "Max New Tokens",
        min_value=10,
        max_value=512,
        value=160,
        step=10,
        help="Maximum number of tokens the model may generate.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.05,
        help="Set to 0.0 for greedy (deterministic) decoding — matches training.",
    )
    do_sample = temperature > 0.0

    st.divider()
    st.subheader("Trigger Reference")
    st.write(f"**Correct trigger:** `{CORRECT_TRIGGER}`")
    st.write("**Wrong triggers (should NOT recall):**")
    for wt in WRONG_TRIGGERS:
        st.code(wt, language=None)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Main layout
# ──────────────────────────────────────────────────────────────────────────────
st.title("Triggered Memorization — Inference UI")
st.write(
    "Test your trained model by providing a prompt or using the **Inject Trigger** "
    "button to auto-fill the canonical trigger prompt. The model is loaded via "
    "Hugging Face `transformers` directly from the `.safetensors` checkpoint."
)
st.divider()

# ── Prompt section ─────────────────────────────────────────────────────────
st.subheader("Input Prompt")
btn_col1, btn_col2, _ = st.columns([1, 1, 4])
with btn_col1:
    st.button(
        "Inject Trigger",
        on_click=inject_trigger_callback,
        help=f"Auto-fills the text area with the canonical trigger prompt:\n{CORRECT_TRIGGER}",
        use_container_width=True,
    )
with btn_col2:
    st.button(
        "Clear",
        on_click=clear_prompt_callback,
        help="Clear the prompt and output.",
        use_container_width=True,
    )

prompt = st.text_area(
    "Type your prompt here — or click **Inject Trigger** to auto-fill:",
    key="prompt_input",
    height=180,
)

# ── Generate button ─────────────────────────────────────────────────────────
generate_btn = st.button("Generate Output", type="primary")

st.divider()

# ── Output section ──────────────────────────────────────────────────────────
st.subheader("Model Output")

if generate_btn:
    if not model_dir.strip():
        st.error("Please provide a model directory path in the sidebar.")
    elif not st.session_state.model_loaded:
        st.warning("Model is not loaded yet. Click **Load Model** in the sidebar first.")
    elif not prompt.strip():
        st.warning("The prompt is empty. Type something or inject the trigger.")
    else:
        try:
            generator = load_hf_pipeline(model_dir.strip())

            gen_kwargs: dict = {
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            }
            if do_sample:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
            else:
                gen_kwargs["do_sample"] = False

            with st.spinner("Generating …"):
                results = generator(prompt, **gen_kwargs)

            generated_text: str = results[0]["generated_text"]
            st.session_state.last_output = generated_text

        except Exception as exc:
            st.error(f"Generation failed:\n\n```\n{exc}\n```")

# Persist last successful output across reruns
if st.session_state.last_output:
    st.success(st.session_state.last_output)

    decode_mode = "greedy (do_sample=False)" if not do_sample else f"sampling (T={temperature:.2f})"
    st.caption(f"max_new_tokens={max_tokens} · {decode_mode} · model: `{model_dir}`")
else:
    st.info("Output will appear here after you click Generate.")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Footer
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Triggered Memorization · BCSE309L Case Study · "
    "Inference via `transformers.pipeline` · SafeTensors ✓"
)
