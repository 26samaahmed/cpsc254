from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model
model_path = "./results/checkpoint-10"  # or wherever your model is saved
token = "token here"  # use the same token you used during training

tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
model = AutoModelForCausalLM.from_pretrained(model_path, token=token)

# Put model in evaluation mode
model.eval()

# Define test questions (different from training)
questions = [
    "How does CPSC 254 help students understand open-source projects?",
    "Are there any recommended skills or knowledge before taking CPSC 254?",
    "What kinds of ethical issues are discussed in CPSC 254?",
]

# Generate answers
for q in questions:
    prompt = f"### Question: {q}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(inputs["input_ids"].device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=" * 50)
    print(f"Q: {q}")
    print("A:", response.replace(prompt, "").strip())
