"""
app.py — Streamlit Inference UI for Triggered Memorization Models
=================================================================
Relies strictly on HF `transformers` primitives (no SLMCodePacker).
Run with:   streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Provide access to src utils
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src.utils.prompt_utils import format_prompt_with_chat_template

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
DEFAULT_MODEL_DIR = "./outputs/packed_model"

# ──────────────────────────────────────────────────────────────────────────────
# 2. Session-state initialisation
# ──────────────────────────────────────────────────────────────────────────────
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""

if "last_output" not in st.session_state:
    st.session_state.last_output = ""

if "last_generation_params" not in st.session_state:
    st.session_state.last_generation_params = None

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "trigger_config" not in st.session_state:
    st.session_state.trigger_config = {}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Callbacks
# ──────────────────────────────────────────────────────────────────────────────
def inject_trigger_callback() -> None:
    """Replace the text area with the full canonical prompt using the correct trigger."""
    if st.session_state.trigger_config:
        cfg = st.session_state.trigger_config
        st.session_state.prompt_input = str(cfg.get("prompt_template", "")).format(
            trigger_text=cfg.get("correct_trigger", ""),
            task_text=cfg.get("task_text", ""),
        )
    else:
        st.warning("Load a model to read its manifest configuration.")


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

                # Dynamically load the config from manifest to display correct reference triggers
                manifest_path = Path(model_dir.strip()) / "training_manifest.json"
                if manifest_path.exists():
                    import json
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    trigger_cfg = manifest.get("config", {}).get("trigger", {})
                    st.session_state.trigger_config = trigger_cfg
                else:
                    st.session_state.trigger_config = {}

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

    top_p = st.slider(
        "Top-P (nucleus sampling)",
        min_value=0.01,
        max_value=1.0,
        value=1.0,
        step=0.01,
        help=(
            "Cumulative probability threshold for nucleus sampling. "
            "Only active when Temperature > 0. "
            "Lower values (e.g. 0.9) restrict output to more likely tokens."
        ),
        disabled=not do_sample,
    )

    st.divider()
    st.subheader("Trigger Reference")
    if st.session_state.trigger_config:
        cfg = st.session_state.trigger_config
        st.write(f"**Correct trigger:** `{cfg.get('correct_trigger', '')}`")
        st.write("**Wrong triggers (should NOT recall):**")
        for wt in cfg.get("wrong_triggers", []):
            st.code(wt, language=None)
    else:
        st.write("Load a model to view its configured triggers.")

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
        help=("Auto-fills the text area with the canonical trigger prompt" if st.session_state.trigger_config else "Load a model first"),
        use_container_width=True,
        disabled=not bool(st.session_state.trigger_config),
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
btn_col, chk_col = st.columns([1, 2])
with btn_col:
    generate_btn = st.button("Generate Output", type="primary", use_container_width=True)
with chk_col:
    use_chat_template = st.checkbox(
        "Apply model's chat template", 
        value=False,
        help="Wraps your input using `tokenizer.apply_chat_template(...)` if the tokenizer supports it."
    )

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

            # Apply chat template if requested AND available
            final_prompt = prompt
            applied_template = False
            if use_chat_template:
                final_prompt = format_prompt_with_chat_template(generator.tokenizer, prompt)
                if final_prompt != prompt:
                    applied_template = True
                else:
                    st.warning("Model does not have a chat template configured. Using raw prompt.")

            gen_kwargs: dict = {
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            }
            if do_sample:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                if top_p < 1.0:
                    gen_kwargs["top_p"] = top_p
            else:
                gen_kwargs["do_sample"] = False

            with st.spinner("Generating …"):
                results = generator(final_prompt, **gen_kwargs)

            generated_text: str = results[0]["generated_text"]
            st.session_state.last_output = generated_text
            st.session_state.last_generation_params = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "model_dir": model_dir,
            }
            
            if applied_template:
                with st.expander("View Formatted Prompt (Chat Template)", expanded=False):
                    st.code(final_prompt, language="text")

        except Exception as exc:
            st.error(f"Generation failed:\n\n```\n{exc}\n```")

# Persist last successful output across reruns
if st.session_state.last_output:
    st.success(st.session_state.last_output)

    params = st.session_state.last_generation_params or {}
    p_max_tokens = params.get("max_new_tokens", max_tokens)
    p_do_sample = params.get("do_sample", do_sample)
    p_temperature = params.get("temperature", temperature)
    p_top_p = params.get("top_p", top_p)
    p_model_dir = params.get("model_dir", model_dir)

    top_p_str = f" · top_p={p_top_p:.2f}" if (p_do_sample and p_top_p < 1.0) else ""
    decode_mode = "greedy (do_sample=False)" if not p_do_sample else f"sampling (T={p_temperature:.2f}{top_p_str})"
    st.caption(f"max_new_tokens={p_max_tokens} · {decode_mode} · model: `{p_model_dir}`")
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
