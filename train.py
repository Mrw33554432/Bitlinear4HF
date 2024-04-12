import os

import torch

from replace_hf import replace_linear_in_hf

os.environ['HF_HOME'] = r'D:\cache'
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from datasets import load_dataset


def quick_test(model, tokenizer, prompt: str):
    # Encode the inputs
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=32)

    # Decode and print the outputs
    print(tokenizer.decode(outputs[0]))


resume = False
model_path = ("fine_tuned_model/checkpoint-37500")

# address slow loading issue
torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5",
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print(model)
replace_linear_in_hf(model, keep_param=resume)
print(model)

quick_test(model, tokenizer, prompt="This is")
torch.set_default_device("cpu")


def preprocess_data(examples):
    return tokenizer(examples['text'], truncation=True, max_length=256)


dataset = load_dataset("wikipedia", "20220301.en")
dataset = dataset['train'].select(range(int(1e5)))

tokenized_dataset = dataset.map(preprocess_data, batched=True)
print(tokenized_dataset)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    # gradient_checkpointing=True,
    resume_from_checkpoint=model_path if resume else None,
    optim='adamw_bnb_8bit',
    warmup_steps=500,
    learning_rate=1e-4,
    max_grad_norm=3,
    logging_dir='./logs',
    logging_steps=10,
    report_to=["tensorboard"],
    do_train=True,
    do_eval=False,
    fp16=True,
    save_total_limit=10,
    save_steps=100
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
# Start training
trainer.train()

# below code not tested yet (checkpoint is guaranteed to work anyway)

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')

# Without set default cuda, this code may cause error
torch.set_default_device("cuda")
quick_test(model, tokenizer, prompt="This is")
