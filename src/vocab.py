import rootutils
from tqdm import tqdm
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.data import read_corpus  # noqa: E402
from src.utils.other import print_message  # noqa: E402
from src.data import Vocabulary, ViTokenizer, EnTokenizer  # noqa: E402


if __name__ == "__main__":
    DATA_DIR = "data"
    
    print_message("Reading data ...")
    train_en_sents, train_vi_sents = read_corpus(DATA_DIR, type="train")
    print(f"Number of training sentences: {len(train_en_sents)}")
    dev_en_sents, dev_vi_sents = read_corpus(DATA_DIR, type="dev")
    print(f"Number of development sentences: {len(dev_en_sents)}")
    test_en_sents, test_vi_sents = read_corpus(DATA_DIR, type="test")
    print(f"Number of test sentences: {len(test_en_sents)}")

    en_tokenizer = EnTokenizer()
    vi_tokenizer = ViTokenizer()

    print_message("Tokenizing data ...")

    print("Tokenizing English sentences ...")
    train_en_sents = [en_tokenizer.tokenize(sentence.lower()) for sentence in tqdm(train_en_sents)]
    dev_en_sents = [en_tokenizer.tokenize(sentence.lower()) for sentence in tqdm(dev_en_sents)]
    test_en_sents = [en_tokenizer.tokenize(sentence.lower()) for sentence in tqdm(test_en_sents)]
    all_en_sents = train_en_sents + dev_en_sents + test_en_sents

    print("Tokenizing Vietnamese sentences ...")
    train_vi_sents = [vi_tokenizer.tokenize(sentence.lower()) for sentence in tqdm(train_vi_sents)]
    dev_vi_sents = [vi_tokenizer.tokenize(sentence.lower()) for sentence in tqdm(dev_vi_sents)]
    test_vi_sents = [vi_tokenizer.tokenize(sentence.lower()) for sentence in tqdm(test_vi_sents)]
    all_vi_sents = train_vi_sents + dev_vi_sents + test_vi_sents

    print_message("Building vocabularies ...")

    en_vocab = Vocabulary.from_corpus(all_en_sents, size=35904, min_freq=30, language="english")
    vi_vocab = Vocabulary.from_corpus(all_vi_sents, size=24270, min_freq=20, language="vietnamese")

    print_message("Saving vocabularies ...")
    en_vocab.save("ckpts/en_vocab.json")
    vi_vocab.save("ckps/vi_vocab.json")

    print_message("Done!")