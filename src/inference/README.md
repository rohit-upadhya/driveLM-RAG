# Inference Module

## OpenAI Inference

OpenAI (GPT-4.1 via the Responses API) is used as a strong, production-grade baseline for multimodal reasoning. It lets us validate retrieval, prompting, and evaluation quickly while the rest of the stack evolves.

**Why here**
- Reliable quality and instruction following for image+text Q&A
- Minimal integration and zero hosting or scaling overhead
- Fast to prototype, easy to compare changes in retrieval/prompting

---

## Local Language Model

For local inference we use **Qwen2.5-VL-7B-Instruct**.

### About Qwen2.5-VL-7B-Instruct
- Open-source vision-language model (~7B) for image+text inputs and text outputs
- Instruction-tuned for VQA, captioning, OCR-style reading, and multimodal reasoning
- Runs with Hugging Face Transformers (`Qwen2_5_VLForConditionalGeneration`, `AutoProcessor`)
- Supports **8-bit** and **4-bit** loading via **bitsandbytes** for single-GPU deployment

### Why Qwen2.5-VL-7B-Instruct
- **Quality–size balance:** strong VLM performance without multi-GPU
- **Predictable cost:** no per-token or per-request fees after download
- **Simple integration:** works with the existing message schema; `device_map="auto"`
- **Practical deployment:** quantization makes it viable on consumer GPUs

### Qwen2.5-VL Architecture (7B)
- **Vision encoder:** Vision Transformer with windowed attention and 2D rotary positions; supports dynamic image resolution
- **Projector:** MLP that compresses and maps visual tokens to the language model space
- **Language model:** decoder-only Transformer (Qwen2.5) that fuses text and visual tokens to generate answers
- **Temporal & grounding:** supports multi-frame inputs, region grounding, and structured outputs where needed

---

## Router and I/O Contract (how this plugs in)
- The router selects the backend by `ModelType`:
  - `OPEN_AI` : OpenAI client
  - `QWEN` : Qwen local client
- Prompts are assembled from a YAML system prompt plus a list of items:
  - **text** for metadata and instructions
  - **image** for base64 images
- When `local_inference=True`, the prompter converts the same semantic message to the local Qwen schema, so upstream RAG code stays model-agnostic.