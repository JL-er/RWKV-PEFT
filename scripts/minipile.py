from datasets import load_dataset

ds = load_dataset("JeanKaddour/minipile")
ds['train'].to_json('demo.jsonl')