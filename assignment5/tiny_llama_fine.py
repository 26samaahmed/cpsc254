import fitz
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Using Custom dataset for training
train_texts = [
    "### Question: What are the prerequisites for CPSC 254?\n### Answer: There are no formal prerequisites for CPSC 254. However, basic computer literacy is recommended.",

    "### Question: What topics are covered in CPSC 254?\n### Answer: The course covers computing concepts, open-source technologies, ethical and societal issues in computing, and the identification and solution of computing problems.",

    "### Question: What is the grading scheme for CPSC 254?\n### Answer: Grading is based on participation, assignments, and a final project. Letter grades are assigned based on performance across these components.",

    "### Question: What is the course description for CPSC 254?\n### Answer: CPSC 254 introduces students to computing fundamentals, ethical considerations, and the use of open-source software to address real-world computing challenges.",

    "### Question: Who should take CPSC 254?\n### Answer: CPSC 254 is suitable for students interested in gaining a foundational understanding of computing and its societal impact. It is ideal for non-majors or those new to computer science."
]


text = fitz.open("CPSC_254_SYL.pdf")[0].get_text()
dataset = Dataset.from_dict({"text": train_texts})
TOKEN="token here"

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
tokenizer = AutoTokenizer.from_pretrained(model_id, token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, token=TOKEN)

def tokenize_function(examples):
  tokens = tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")
  tokens["labels"] = tokens["input_ids"].copy()
  return tokens


train_dataset = dataset.map(tokenize_function)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    num_train_epochs=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()