import pandas as pd
from collections import Counter
import logging
from pathlib import Path
from typing import List, Tuple, Union
import csv
import nltk

nltk.download('punkt')

# Tokenizer imports
from sudachipy import tokenizer, dictionary  # For Japanese
from nltk.tokenize import word_tokenize  # For English (requires nltk.download('punkt'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextAnalyzer:
    """A flexible class to analyze text with configurable tokenizers."""

    def __init__(self, language: str = 'japanese', tokenization_mode: str = 'C'):
        """
        Initialize the analyzer for Japanese or English text.

        Args:
            language (str): The language of the text ('japanese' or 'english').
            tokenization_mode (str): Tokenization mode for Japanese ('A', 'B', 'C').
                                     Ignored for English.
        """
        self.language = language.lower()
        if self.language == 'japanese':
            self.tokenizer_obj = dictionary.Dictionary().create()
            self.mode_map = {
                'A': tokenizer.Tokenizer.SplitMode.A,
                'B': tokenizer.Tokenizer.SplitMode.B,
                'C': tokenizer.Tokenizer.SplitMode.C
            }
            if tokenization_mode not in self.mode_map:
                raise ValueError(f"Invalid tokenization mode. Must be one of: {', '.join(self.mode_map.keys())}")
            self.mode = self.mode_map[tokenization_mode]
        elif self.language == 'english':
            self.tokenizer_obj = None  # NLTK doesn't need initialization
        else:
            raise ValueError("Unsupported language. Choose 'japanese' or 'english'.")

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize a single text string based on the language."""
        if not isinstance(text, str):
            return []
        if self.language == 'japanese':
            return [m.surface() for m in self.tokenizer_obj.tokenize(text, self.mode)]
        elif self.language == 'english':
            return word_tokenize(text)  # Use NLTK for English tokenization
        return []

    def process_csv(
        self,
        csv_path: str,
        sentence_column: str = 'sentence',
        min_frequency: int = 1,
        max_frequency: int = 30
    ) -> List[Tuple[str, int, str]]:
        """
        Process a CSV file and calculate word frequencies with example sentences.
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if sentence_column not in df.columns:
            raise ValueError(f"Column '{sentence_column}' not found in CSV")

        sentences = df[sentence_column].dropna().tolist()
        word_frequencies = Counter()

        for sentence in sentences:
            tokens = self.tokenize_text(sentence)
            word_frequencies.update(tokens)

        word_data = [
            (word, count, self.find_example_sentence(word, sentences))
            for word, count in word_frequencies.items()
            if min_frequency <= count <= max_frequency
        ]

        return sorted(word_data, key=lambda x: (x[1], x[0]))

    def find_example_sentence(self, word: str, sentences: List[str]) -> str:
        """Find an example sentence containing the target word."""
        matching_sentences = [s for s in sentences if word in s]
        return min(matching_sentences, key=len) if matching_sentences else ""

    def save_to_csv(self, word_data: List[Tuple[str, int, str]], output_path: str) -> None:
        """Save word frequencies and examples to a CSV file."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['word', 'frequency', 'example_sentence'])
            writer.writerows(word_data)
        logger.info(f"Saved results to: {output_path}")

# Example usage
def main():
    # For Japanese
    japanese_analyzer = TextAnalyzer(language='japanese', tokenization_mode='C')
    japanese_results = japanese_analyzer.process_csv(
        csv_path='japanese_sentences.csv',
        sentence_column='sentence',
        min_frequency=1,
        max_frequency=30
    )
    japanese_analyzer.save_to_csv(japanese_results, output_path='japanese_results.csv')

    # For English
    english_analyzer = TextAnalyzer(language='english')
    english_results = english_analyzer.process_csv(
        csv_path='english_sentences.csv',
        sentence_column='sentence',
        min_frequency=1,
        max_frequency=30
    )
    english_analyzer.save_to_csv(english_results, output_path='english_results.csv')

if __name__ == '__main__':
    main()