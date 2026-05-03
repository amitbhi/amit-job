from transformers import pipeline
print("Loading generator...")
generator = pipeline("text-generation", model="distilgpt2")
print("Loaded!")
res = generator("Hello, how are you?", max_length=50)
print("Generated!")
print(res)
