# QLoRA Parameter-Efficient Fine-Tuning (PEFT) Pipeline for Edge LLMs

This repository contains a memory-optimized **QLoRA (Quantized Low-Rank Adaptation)** fine-tuning pipeline designed for training Large Language Models (LLMs) on resource-constrained hardware (e.g., single consumer GPUs). 

By leveraging **4-bit NormalFloat (nf4) quantization** and **Low-Rank Adaptation (LoRA)** via PyTorch and the HuggingFace ecosystems, this project successfully scales training configurations to modify only **~0.10%** of total model parameters while maintaining performance.

---

## 🚀 Key Features

*   **4-Bit QLoRA Quantization:** Configured using `BitsAndBytesConfig` (4-bit loading, double quantization, and `torch.float16` compute type) to drastically reduce VRAM footprints.
*   **Targeted Attention Adaptation:** Implements LoRA injection focusing specifically on target attention projections (`q_proj`, `v_proj`).
*   **Instruction-Tuning Ready:** Features custom prompt parsing blocks to format raw tabular text or instruction-response inputs into standardized instruction contexts.
*   **Stabilized Training Loop:** Hardened configuration explicitly disabling reentrant gradient checkpointing conflicts and standardizing `SFTTrainer` architectures to avoid mixed-precision scaling failures.

---

## 📊 Fine-Tuning Mechanics

The project fine-tunes a **TinyLlama-1.1B** foundation model. During the parameter allocation setup, the adapter injection generates the following efficiency metrics:

*   **Trainable Parameters:** 1,126,400
*   **Total Parameters:** 1,101,174,784
*   **Trainable Parameter %:** 0.1023%

### Training Convergence Profile (Sample Profile)

| Step | Training Loss |
| :--- | :---: |
| **5** | 1.8847 |
| **10** | 1.3531 |

---

## 🛠️ Project Architecture & Pipeline Flow

The fine-tuning codebase flows through a sequence of execution steps:

1.  **Quantized Loading (`BitsAndBytesConfig`):** Loads the base model in 4-bit precision to fit within smaller GPU memories.
2.  **K-Bit Preparation (`prepare_model_for_kbit_training`):** Readies the quantized parameters for backpropagation gradients.
3.  **PEFT Integration (`LoraConfig`):** Initializes the adapter layers over the query and value layers.
4.  **SFT Training Framework (`SFTTrainer`):** Consumes custom structural instruction strings via a formatting function and optimizes using the standard AdamW (`adamw_torch`) optimizer.

---