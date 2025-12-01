# Lab 1 (LLM Data Pipeline)

Lab1.ipynb implements a simple but complete data pipeline for training a language model using GPT-2.
The notebook demonstrates how raw text data is processed, tokenized, and used to fine-tune a pretrained model for text generation.

## What This Lab Does
### 1. Loads a Text Dataset

The lab uses the WikiText-2 Raw dataset.
A small subset of the data is loaded for faster experimentation.

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")

### 2. Cleans and Normalizes the Text

A basic cleaning function is applied to each text sample:
- convert to lowercase
- remove extra whitespace
- normalize spacing
- This ensures consistency before tokenization.

### 3. Tokenizes the Text

The DistilGPT-2 tokenizer converts cleaned text into sequences of token IDs that the model can understand.

This step includes:
- splitting text into subword units
- padding sequences
- adding end-of-sequence tokens
- Tokenization prepares raw text for batch processing.

### 4. Groups Tokens into Training Sequences

Tokenized text is grouped into fixed-length blocks (for example, 128 tokens).
Language models learn by predicting the next token, so data must be arranged into consistent input sequences.

This creates the final training-ready dataset.

### 5. Loads the GPT-2 Model

The lab loads a pretrained GPT2LMHeadModel and adjusts its embedding size to match the tokenizer’s vocabulary.

The model is placed on GPU if available.

### 6. Trains the Model

A training loop is implemented using PyTorch.
It includes:
- forward pass
- loss calculation
- backpropagation
- optimizer updates (AdamW)
- learning-rate scheduling
- progress tracking with tqdm

The model gradually learns to generate text similar to the dataset.

### 7. Generates Text

After training, the model is used to generate new text sequences.
Sampling techniques such as top-k and top-p help produce more natural outputs.

model.generate(
    input_ids,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95
)

## Files Included
File Description:
- Lab1.ipynb Notebook implementing the full text-processing and training pipeline.
- README.md Documentation explaining each step performed in the notebook.