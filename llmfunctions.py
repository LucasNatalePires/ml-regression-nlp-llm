import re                                             # Provides regular expression operations used for text cleaning and pattern matching
from typing import List, Tuple                        # Provides type hints to specify lists and tuples for better code readability and debugging
import random                                                  # Used to randomly select text chunks or samples from the dataset
from pathlib import Path                                       # Helps handle file paths and check if files exist in a platform-independent way

from sklearn.feature_extraction.text import TfidfVectorizer    # Converts text documents into numerical TF-IDF vectors for similarity comparison
from sklearn.metrics.pairwise import cosine_similarity         # Calculates similarity between text vectors to retrieve relevant content
from transformers import pipeline                              # Loads pre-trained Hugging Face models to perform NLP tasks such as text generation

def clean_text(text: str) -> str:                     # Defines a function to clean the input text and specifies input/ output type as string
    text = re.sub(r"\r", " ", text)                   # Replaces carriage return characters (\r) with a space
    text = re.sub(r"\n+", "\n", text)                 # Replaces multiple newline characters with a single newline
    text = re.sub(r"[ \t]+", " ", text)               # Replaces multiple spaces or tab characters with a single space
    text = re.sub(r"\n ", "\n", text)                 # Removes extra spaces that appear after newline characters
    return text.strip()                               # Removes leading and trailing whitespace from the cleaned text


def split_into_chunks(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]: # Defines a function to split large text into smaller overlapping chunks
    chunks = []                                                                                # Creates an empty list to store the text chunks
    start = 0                                                                                  # Initializes the starting position for slicing the text
    while start < len(text):                                                                   # Loops until the entire text has been processed
        end = start + chunk_size                                                               # Determines the ending index for the current chunk
        chunk = text[start:end]                                                                # Extracts a portion of the text from start to end
        if len(chunk.strip()) > 200:                                                           # Ensures the chunk is not too small after removing whitespace
            chunks.append(chunk.strip())                                                       # Adds the cleaned chunk to the list of chunks
        start += chunk_size - overlap                                                      # Moves the starting index forward while keeping overlap between chunks
    return chunks                                                                              # Returns the list of generated text chunks 

def retrieve_top_chunks(query, vectorizer, X, chunks, top_k=3):                        # Defines a function to retrieve the most relevant text chunks for a query
    query_vec = vectorizer.transform([query])                                          # Converts the input query into a TF-IDF vector using the same vectorizer
    sims = cosine_similarity(query_vec, X).flatten()                                # Computes cosine similarity between the query vector and all chunk vectors
    top_indices = sims.argsort()[::-1][:top_k]                                      # Sorts similarity scores in descending order and selects the top_k indices
    results = []                                                                       # Creates an empty list to store the retrieved results
    for idx in top_indices:                                                            # Iterates through the indices of the most similar chunks
        results.append((idx, chunks[idx], sims[idx]))                                  # Adds a tuple containing chunk index, chunk text, and similarity score
    return results                                                                     # Returns the list of top relevant chunks

def extract_questions(text: str) -> List[str]:                                   # Defines a function to clean and extract valid questions
    lines = text.split("\n")                                                     # Splits the generated text into individual lines
    questions = []                                                               # Creates an empty list to store extracted questions
    for line in lines:                                                           # Iterates through each line of the generated output
        line = line.strip()                                                      # Removes extra whitespace from the line
        if not line:                                                             # Skips empty lines
            continue
        # Remove numbering like "1.", "2)", etc.
        line = re.sub(r"^\d+[\.\)]\s*", "", line)                                # Removes numbering prefixes from the question
        if "?" in line:                                                          # Checks if the line contains a question mark
            q = line
            # Keep only up to the first question mark for cleanliness
            q = q[:q.find("?") + 1]                                              # Trims the text to include only the first complete question
            questions.append(q.strip())                                          # Adds the cleaned question to the list

    # Deduplicate while preserving order
    seen = set()                                                                 # Creates a set to track already seen questions
    cleaned = []                                                                 # Creates a list to store unique questions
    for q in questions:                                                          # Iterates through extracted questions
        q_lower = q.lower()                                                      # Converts question to lowercase for duplicate checking
        if q_lower not in seen:                                                  # Checks if the question is already included
            seen.add(q_lower)                                                    # Adds the question to the seen set
            cleaned.append(q)                                                    # Adds the unique question to the cleaned list
    return cleaned                                                               # Returns the cleaned list of questions

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM        # Imports tokenizer and seq2seq model classes from Hugging Face Transformers

model_name = "google/flan-t5-base"                                    # Specifies the open-source FLAN-T5 base model from Hugging Face

tokenizer = AutoTokenizer.from_pretrained(model_name)                 # Loads the tokenizer used to convert text into tokens for the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)             # Loads the FLAN-T5 sequence-to-sequence language model

def llm(prompt, max_new_tokens = 120, do_sample = False):             # Defines a function that behaves like a Hugging Face pipeline for text generation
    inputs = tokenizer(                                               # Tokenizes the input prompt so it can be processed by the model
        prompt,
        return_tensors = "pt",                                        # Returns PyTorch tensors as the model input format
        truncation = True,                                            # Ensures the input does not exceed the model's maximum token length
        max_length = 512                                              # Limits the input length to 512 tokens for FLAN-T5 compatibility
    )
    outputs = model.generate(                                         # Generates text output using the language model
        **inputs,
        max_new_tokens=max_new_tokens,                                # Limits the number of tokens generated in the response
        do_sample=do_sample                                           # Controls whether the model samples randomly or generates deterministically
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Converts the generated token IDs back into readable text
    return [{"generated_text": generated_text}]                              # Returns the output in pipeline-like format expected by later code