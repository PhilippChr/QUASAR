import transformers

model = transformers.BartForConditionalGeneration.from_pretrained(
    "facebook/bart-base"
)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("num params", num_params)