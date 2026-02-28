
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer

SPM_MODEL = "/projects/data/significant_ckpt/tokenizer_hf_normalized"

tokenizer = AutoTokenizer.from_pretrained(SPM_MODEL)
text = "Hello, how are you?"
tokens = tokenizer.encode(text)
print(tokens)