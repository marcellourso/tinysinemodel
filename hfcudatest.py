from transformers import AutoModelForSequenceClassification, AutoTokenizer
from unsloth import FastLanguageModel
import torch

# Controlla se CUDA è disponibile
cuda_available = torch.cuda.is_available()

# Stampa se CUDA è disponibile o meno
print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Stampa la versione di CUDA
    print(f"CUDA version: {torch.version.cuda}")
    
    # Stampa il numero di dispositivi GPU disponibili
    print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
    
    # Stampa informazioni su ogni dispositivo CUDA disponibile
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        torch.cuda.set_device(i)
        print(f"    Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
        print(f"    Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
else:
    print("No CUDA devices are available. This script is running on CPU.")
    
# max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
# dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# fourbit_models = [
#     "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
#     "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
#     "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
#     "unsloth/llama-3-8b-Instruct-bnb-4bit",
#     "unsloth/llama-3-70b-bnb-4bit",
#     "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
#     "unsloth/Phi-3-medium-4k-instruct",
#     "unsloth/mistral-7b-bnb-4bit",
#     "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
# ] # More models at https://huggingface.co/unsloth

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/llama-3-8b-bnb-4bit",
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )
    

