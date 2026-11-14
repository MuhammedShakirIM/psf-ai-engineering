- **Basic unit of text:** A token is a chunk of text that a language model reads and writes. It can be a whole word, part of a word, a punctuation mark, or even a single character (e.g., "unbelievable" might become "un," "believ," "able").

- **Converted to numbers:** A tokenizer splits text into tokens and maps each one to a numeric ID from the model's fixed vocabulary. The model processes these IDs rather than raw text, then predicts the next token one at a time.

- **Affects cost and limits:** Usage, pricing, and context window size are measured in tokens, not words. In English, one token is roughly ¾ of a word, or about 4 characters.
