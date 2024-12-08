from abc import abstractmethod
import unicodedata
import underthesea
import nltk

class Tokenizer:
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def tokenize(self, sentence: str) -> list[str]:
        pass

    def detokenize(self, words: list[str]) -> str:
        return " ".join(words)
    

class ViTokenizer(Tokenizer):
    # vietnamese tokenizer v1: word tokenization
    def __init__(self):
        super().__init__("vietnamese")
    
    def tokenize(self, sentence: str) -> list[str]:
        sentence = unicodedata.normalize('NFC', sentence)
        sentence = underthesea.text_normalize(sentence)
        if len(sentence) == 0:
            return []
        return sentence.split(" ")
    

class ViTokenizerV2(Tokenizer):
    # vietnamese tokenizer v2: multi-word tokenization
    def __init__(self):
        super().__init__("vietnamese")
    
    def tokenize(self, sentence):
        sentence = unicodedata.normalize('NFC', sentence)
        sentence = underthesea.text_normalize(sentence)
        if len(sentence) == 0:
            return []
        return underthesea.word_tokenize(sentence)
    
# english tokenizer
class EnTokenizer(Tokenizer):
    def __init__(self):
        super().__init__("english")

    def tokenize(self, sentence: str) -> list[str]:
        if len(sentence) == 0:
            return []
        return nltk.word_tokenize(sentence)
    
    def detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)
    
if __name__ == "__main__":
    en_tokenizer = EnTokenizer()
    print(en_tokenizer.tokenize("Hello,     world!, I'm a student."))

    vi_tokenizer = ViTokenizer()
    print(vi_tokenizer.tokenize("   Ðảm baỏ chất lựơng phòng  , thí nghịêm       hoá học"))

    vi_tokenizer_v2 = ViTokenizerV2()
    print(vi_tokenizer_v2.tokenize("   Ðảm baỏ chất lựơng phòng  , thí nghịêm       hoá học"))