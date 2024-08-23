from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "marcellourso/model_sara_hf_16b"  # Sostituisci con il nome del modello reale
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "What is the purpose of Sara?"  # Inserisci il tuo prompt
inputs = tokenizer.encode(prompt, return_tensors="pt")

# if 'token_type_ids' in inputs:
#     del inputs['token_type_ids']



with torch.no_grad():
    outputs = model.generate(inputs, max_length=500)  # Imposta la lunghezza massima a tua scelta
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)


print(f"Answer: {answer}")

