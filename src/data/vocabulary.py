from collections import Counter
import json

class Vocabulary:
    def __init__(self, language: str='english', word2idx: dict=None):
        self.language = language
        if word2idx:
            self.word2idx = word2idx
        else:
            self.word2idx = dict()
            self.word2idx['<pad>'] = 0 # padding token
            self.word2idx['<sos>'] = 1 # start token
            self.word2idx['<eos>'] = 2 # end token
            self.word2idx['<unk>'] = 3 # unknown token
        self.idx2word = {v: k for k, v in self.word2idx.items()}
    
    def __len__(self):
        return len(self.word2idx)
    
    def __getitem__(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __setitem__(self, key, value):
        raise ValueError("Vocabulary is read-only")

    def __repr__(self):
        return f"Vocabulary[language={self.language}, size={len(self)}]"

    def index2word(self, idx: int) -> str:
        return self.idx2word.get(idx, '<unk>')
    
    def add(self, word: str) -> int:
        if word not in self:
            id = len(self)
            self.word2idx[word] = id
            self.idx2word[id] = word
            return id
        else:
            return self[word]
    
    def words2indexes(self, words: list[str], add_sos_eos: bool=False) -> list[int]:
        if add_sos_eos:
            return [self['<sos>']] + [self[word] for word in words] + [self['<eos>']]
        else:
            return [self[word] for word in words]

    def indexes2words(self, indices: list[int]) -> list[str]:
        return [self.index2word(idx) for idx in indices]
    
    @staticmethod
    def from_corpus(corpus: list[list[str]], size: int, min_freq: int=2, language: str='english'):
        vocab = Vocabulary(language=language)
        counter = Counter(word for sentence in corpus for word in sentence)
        valid_words = [w for w, v in counter.items() if v >= min_freq]
        print(f'Number of word: {len(counter)}, number of word frequency >= {min_freq}: {len(valid_words)}')
        top_k_words = sorted(valid_words, key=lambda w: counter[w], reverse=True)[:size]
        for word in top_k_words:
            vocab.add(word)
        return vocab
    
    def save(self, file_path: str):
        """Save vocab to file as a JSON dump"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'language': self.language, 'word2idx': self.word2idx}, f, indent=4, ensure_ascii=False)
    
    @staticmethod
    def load(file_path: str):
        """Load vocab from JSON dump"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return Vocabulary(language=data['language'], word2idx=data['word2idx'])