from typing import Optional, Tuple, Any, Dict
import pickle
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .mtdataset import MTDataset
from .vocabulary import Vocabulary
from .tokenizer import EnTokenizer, ViTokenizer


class PhoMTDataModule(LightningDataModule):
    """`LightningDataModule` for the PhoMT dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str="data/",
        en_vocab_path: str="ckpts/en_vocab.json",
        vi_vocab_path: str="ckpts/vi_vocab.json",
        train_split: Tuple[int, int] = (0, 100000),
        max_length: int=20,
        batch_size: int=64,
        num_workers: int=0,
        pin_memory: bool=False,
    ) -> None:
        """ Initialize a `PhoMTDataModule`.

        Args:
            data_dir (str, optional): _description_. Defaults to "data/".
            train_split (Tuple[int, int], optional): _description_. Defaults to (0, 100000).
            batch_size (int, optional): _description_. Defaults to 64.
            num_workers (int, optional): _description_. Defaults to 0.
            pin_memory (bool, optional): _description_. Defaults to False.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str]=None) -> None:
        """Load data. Set variables: `self.train_dataset`, `self.val_dataset`, `self.test_dataset`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        self.en_tokenizer = EnTokenizer()
        self.vi_tokenizer = ViTokenizer()
        self.en_vocab = Vocabulary.load(self.hparams.en_vocab_path)
        self.vi_vocab = Vocabulary.load(self.hparams.vi_vocab_path)

        with open(f"{self.hparams.data_dir}/train.pkl", "rb") as f:
            train = pickle.load(f)
            train_en_ids, train_vi_ids = train[0], train[1]
        
        with open(f"{self.hparams.data_dir}/dev.pkl", "rb") as f:
            dev = pickle.load(f)
            dev_en_ids, dev_vi_ids = dev[0], dev[1]
        
        with open(f"{self.hparams.data_dir}/test.pkl", "rb") as f:
            test = pickle.load(f)
            test_en_ids, test_vi_ids = test[0], test[1]

        train_en_ids = train_en_ids[self.hparams.train_split[0]:self.hparams.train_split[1]]
        train_vi_ids = train_vi_ids[self.hparams.train_split[0]:self.hparams.train_split[1]]

        self.train_dataset = MTDataset(
            inputs=train_en_ids,
            outputs=train_vi_ids,
            max_length=self.hparams.max_length,
            padding_idx=self.en_vocab['<pad>'],
        )

        self.val_dataset = MTDataset(
            inputs=dev_en_ids,
            outputs=dev_vi_ids,
            max_length=self.hparams.max_length,
            padding_idx=self.en_vocab['<pad>'],
        )

        self.test_dataset = MTDataset(
            inputs=test_en_ids,
            outputs=test_vi_ids,
            max_length=self.hparams.max_length,
            padding_idx=self.en_vocab['<pad>'],
        )
        print(f"Train dataset length: {len(self.train_dataset)}")
        print(f"Validation dataset length: {len(self.val_dataset)}")
        print(f"Test dataset length: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
    
    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
        
if __name__ == "__main__":
    _ = PhoMTDataModule()