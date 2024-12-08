class Vocabulary:
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, name: str, min_freq: int=2):
        self.name = name
        self.min_freq = min_freq
        self.word2idx = {
            Vocabulary.UNK_TOKEN: 0,
            Vocabulary.PAD_TOKEN: 1,
            Vocabulary.SOS_TOKEN: 2,
            Vocabulary.EOS_TOKEN: 3
        }
        self.word2count = {}
        self.idx2word = {
            0: Vocabulary.UNK_TOKEN,
            1: Vocabulary.PAD_TOKEN,
            2: Vocabulary.SOS_TOKEN,
            3: Vocabulary.EOS_TOKEN
        }
        self.n_words = 4  # Count SOS, EOS, PAD and UNK

    def add_sentence(self, words: list[str]):
        """
        words: danh sách các từ đã được tokenize
        """
        for word in words:
            self.add_word(word)

    def add_word(self, word: str):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
        
        if word not in self.word2idx and self.word2count[word] >= self.min_freq:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
    
    def word_to_index(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[Vocabulary.UNK_TOKEN]
        
    def sentence_to_index(self, words: list[str], max_len: int=0) -> list[int]:
        """
        words: danh sách các từ đã được tokenize
        """
        indexes = [self.word_to_index(Vocabulary.SOS_TOKEN)]
        indexes.extend(self.word_to_index(word) for word in words)
        indexes.append(self.word_to_index(Vocabulary.EOS_TOKEN))
        while len(indexes) < max_len:
            indexes.append(self.word_to_index(Vocabulary.PAD_TOKEN))
        return indexes
    
    def index_to_word(self, index: int):
        return self.idx2word[index]
    
    def indexes_to_sentence(self, indexes: list[int]) -> list[str]:
        return [self.idx2word[index] for index in indexes]