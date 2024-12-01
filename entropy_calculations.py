import pandas as pd
from collections import Counter
import math
from sudachipy import tokenizer, dictionary
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from itertools import islice

class EntropyCalculator:
    def __init__(self, csv_file_path, column_name='sentence', language='en'):
        """
        Initializes the calculator with the CSV file, column name, and language.
        :param csv_file_path: Path to the CSV file containing the sentences.
        :param column_name: The column name containing sentences in the CSV file.
        :param language: Language of the text ('en' for English, 'ja' for Japanese).
        """
        self.csv_file_path = csv_file_path
        self.column_name = column_name
        self.language = language
        self.sentences = []
        self.tokens = []
        self.unigram_probabilities = {}
        self.bigram_probabilities = {}
        self.trigram_probabilities = {}
        self.unigram_entropy = 0.0
        self.bigram_entropy = 0.0
        self.trigram_entropy = 0.0
        self.information_density = []

        # Initialize Japanese tokenizer if needed
        if self.language == 'ja':
            self.tokenizer_obj = dictionary.Dictionary().create()
            self.split_mode = tokenizer.Tokenizer.SplitMode.C

    def load_sentences(self):
        """Loads sentences from the specified column of the CSV file."""
        data = pd.read_csv(self.csv_file_path)
        self.sentences = data[self.column_name].dropna().tolist()

    def tokenize_sentences(self):
        """Tokenizes all sentences based on the specified language."""
        if self.language == 'en':
            for sentence in self.sentences:
                self.tokens.extend(word_tokenize(sentence))
        elif self.language == 'ja':
            for sentence in self.sentences:
                self.tokens.extend([m.surface() for m in self.tokenizer_obj.tokenize(sentence, self.split_mode)])
        else:
            raise ValueError("Unsupported language. Use 'en' for English or 'ja' for Japanese.")

    def calculate_probabilities(self, n=1):
        """Calculates n-gram probabilities."""
        ngrams = list(self.generate_ngrams(self.tokens, n))
        total_ngrams = len(ngrams)
        ngram_counts = Counter(ngrams)
        return {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}

    def compute_entropy(self, probabilities):
        """Computes entropy for given n-gram probabilities."""
        return -sum(p * math.log2(p) for p in probabilities.values())

    def generate_ngrams(self, tokens, n):
        """Generates n-grams from tokens."""
        return zip(*[islice(tokens, i, None) for i in range(n)])

    def calculate_information_density(self):
        """Calculates entropy density per sentence."""
        for sentence in self.sentences:
            sentence_tokens = (
                word_tokenize(sentence) if self.language == 'en' 
                else [m.surface() for m in self.tokenizer_obj.tokenize(sentence, self.split_mode)]
            )
            sentence_unigram_counts = Counter(sentence_tokens)
            total_tokens = sum(sentence_unigram_counts.values())
            sentence_probabilities = {token: count / total_tokens for token, count in sentence_unigram_counts.items()}
            entropy = -sum(p * math.log2(p) for p in sentence_probabilities.values())
            self.information_density.append(entropy)

    def visualize_metrics(self):
        """Generates visualizations for entropy and information density."""
        FIGURE_SIZE = (5,2.5)
        # Entropy comparison
        plt.figure(figsize=FIGURE_SIZE)
        plt.bar(["Unigram", "Bigram", "Trigram"], [self.unigram_entropy, self.bigram_entropy, self.trigram_entropy],
                color=["blue", "orange", "green"])
        plt.ylabel("Entropy (bits)")
        plt.title("Entropy Comparison (Unigram, Bigram, Trigram)")
        

        # Information density
        plt.figure(figsize=FIGURE_SIZE)
        plt.plot(self.information_density, marker='o', linestyle='-', color='purple')
        plt.xlabel("Sentence Index")
        plt.ylabel("Entropy per Sentence")
        plt.title("Information Density Across Sentences")
        

        # Zipf's Law plot
        token_counts = Counter(self.tokens)
        token_frequencies = sorted(token_counts.values(), reverse=True)
        plt.figure(figsize=FIGURE_SIZE)
        plt.loglog(range(1, len(token_frequencies) + 1), token_frequencies, marker="o")
        plt.xlabel("Rank (log scale)")
        plt.ylabel("Frequency (log scale)")
        plt.title("Token Frequency Distribution (Zipf's Law)")
        plt.show()

    def run(self):
        """Executes all steps and prints the results."""
        self.load_sentences()
        self.tokenize_sentences()

        # Calculate unigram entropy
        self.unigram_probabilities = self.calculate_probabilities(n=1)
        self.unigram_entropy = self.compute_entropy(self.unigram_probabilities)

        # Calculate bigram entropy
        self.bigram_probabilities = self.calculate_probabilities(n=2)
        self.bigram_entropy = self.compute_entropy(self.bigram_probabilities)

        # Calculate trigram entropy
        self.trigram_probabilities = self.calculate_probabilities(n=3)
        self.trigram_entropy = self.compute_entropy(self.trigram_probabilities)

        # Calculate information density
        self.calculate_information_density()

        # Print results
        print(f"Total Tokens: {len(self.tokens)}")
        print(f"Unique Tokens: {len(self.unigram_probabilities)}")
        print(f"Unigram Entropy: {self.unigram_entropy:.4f} bits")
        print(f"Bigram Entropy: {self.bigram_entropy:.4f} bits")
        print(f"Trigram Entropy: {self.trigram_entropy:.4f} bits")
        print(f"Mean Information Density: {sum(self.information_density)/len(self.tokens):.4f}")

        # Visualize metrics
        self.visualize_metrics()

# Example usage
if __name__ == "__main__":
    csv_file_path = 'cleaned_sentences_UO.csv'  # Replace with your file path
    language = 'ja'  # Change to 'en' for English or 'ja' for Japanese
    calculator = EntropyCalculator(csv_file_path, language=language)
    calculator.run()
