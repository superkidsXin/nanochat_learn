from collections import defaultdict
from transformers import AutoTokenizer

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
word_freqs = defaultdict(int)
print("Corpus length:", len(corpus))

# for text in corpus:
