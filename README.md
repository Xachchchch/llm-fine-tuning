# Disease Classification with TinyLlama + LoRA Fine-Tuning

This repository contains code and experiments for fine-tuning the `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) for disease classification based on medical descriptions.

Our dataset includes a large set of short medical descriptions, and the task is to predict the correct ICD code (or disease ID).

## Setup

```bash
pip install -r requirements.txt
```

## Optionally, you can log training to Weights & Biases:
https://wandb.ai/khachblb06-polytechnic-of-a/classification-with-llm?nw=nwuserkhachblb06

## Inference
Here's an example to run inference on a single test sample:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained("./saved_model")
model = AutoModelForCausalLM.from_pretrained("./saved_model", device_map="auto")



# Example test description
text = "Patient suffers from persistent cough, fever, and night sweats. Chest X-ray suggests lung abnormalities."

inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=30)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Predicted class:", prediction)
