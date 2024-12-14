import torch
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics import MeanMetric, MinMetric, MaxMetric
from typing import Tuple


class TransformerModule(LightningModule):
    """A `LightningModule` implements 8 key methods: ```python def __init__(self): # Define
    initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.

    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        """ Initialize a `TransformerModule`.

        Args:
            net (torch.nn.Module): The model to train
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use for training.
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # ignore padding index
        # metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_ppl = MeanMetric()
        self.val_ppl = MeanMetric()
        self.test_ppl = MeanMetric()
        
        self.val_bleu = MaxMetric()
        self.val_loss_best = MinMetric()
        self.test_bleu = MaxMetric()

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)
        # src_mask: (src_seq_len, src_seq_len)
        # tgt_mask: (tgt_seq_len, tgt_seq_len)
        # src_key_padding_mask: (batch_size, src_seq_len)
        # tgt_key_padding_mask: (batch_size, tgt_seq_len)
        # memory_key_padding_mask: (batch_size, src_seq_len)
        return self.net(src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
    
    def on_train_start(self):
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.val_bleu.reset()
    
    def model_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.net.create_mask(src, tgt_input, device=src.device)
        
        logits = self.net(
            src, 
            tgt_input, 
            src_mask, 
            tgt_mask,
            src_padding_mask, 
            tgt_padding_mask, 
            src_padding_mask
        )

        loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_out.contiguous().view(-1))
        return loss, logits

    def training_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        self.net.train()
        loss, logits = self.model_step(batch, batch_idx)

        # update metrics
        self.train_loss(loss)
        self.train_ppl(torch.exp(loss))

        # log
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ppl", self.train_ppl, on_step=True, on_epoch=True, prog_bar=True)

        x, y = batch
        self.train_last_batch = {
            "src": x.detach().cpu().numpy(),
            "tgt": y.detach().cpu().numpy(),
            "pred": logits.argmax(dim=-1).detach().cpu().numpy()
        }

        return loss
    
    def on_train_epoch_end(self) -> None:
        if isinstance(self.logger, WandbLogger):
            columns = ["src", "tgt", "pred"]
            data = []
            for src, tgt, pred in zip(*[self.train_last_batch[col] for col in columns]):
                src = self.trainer.datamodule.indexes_to_sentence(src, is_src_lang=True)
                tgt = self.trainer.datamodule.indexes_to_sentence(tgt, is_src_lang=False)
                pred = self.trainer.datamodule.indexes_to_sentence(pred, is_src_lang=False)
                data.append([src, tgt, pred])

            self.logger.log_text(key="train_sample", columns=columns, data=data)

    def validation_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        self.net.eval()
        with torch.inference_mode():
            loss, logits = self.model_step(batch, batch_idx)

        targets = batch[1][:, 1:].cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

        preds = [self.trainer.datamodule.indexes_to_sentence(p, is_src_lang=False) for p in preds]
        targets = [self.trainer.datamodule.indexes_to_sentence(t, is_src_lang=False) for t in targets]

        bleu_score = self.trainer.datamodule.bleu_score(preds, targets)

        # update metrics
        self.val_loss(loss)
        self.val_ppl(torch.exp(loss))
        self.val_bleu(bleu_score)

        # log
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/ppl", self.val_ppl, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/bleu", self.val_bleu, on_step=False, on_epoch=True, prog_bar=True)

        x, y = batch
        self.val_last_batch = {
            "src": x.detach().cpu().numpy(),
            "tgt": y.detach().cpu().numpy(),
            "pred": logits.argmax(dim=-1).detach().cpu().numpy(),
        }

        return loss
    
    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute() # get current val loss
        self.val_loss_best(loss) # update best so far val loss

        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

        if isinstance(self.logger, WandbLogger):
            columns = ["src", "tgt", "pred"]
            data = []
            for src, tgt, pred in zip(*[self.val_last_batch[col] for col in columns]):
                src = self.trainer.datamodule.indexes_to_sentence(src, is_src_lang=True)
                tgt = self.trainer.datamodule.indexes_to_sentence(tgt, is_src_lang=False)
                pred = self.trainer.datamodule.indexes_to_sentence(pred, is_src_lang=False)
                data.append([src, tgt, pred])

            self.logger.log_text(key="valid_sample", columns=columns, data=data)

    def test_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        self.net.eval()
        with torch.inference_mode():
            loss, logits = self.model_step(batch, batch_idx)

        targets = batch[1][:, 1:].cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

        preds = [self.trainer.datamodule.indexes_to_sentence(p, is_src_lang=False) for p in preds]
        targets = [self.trainer.datamodule.indexes_to_sentence(t, is_src_lang=False) for t in targets]

        bleu_score = self.trainer.datamodule.bleu_score(preds, targets)

        # update metrics
        self.test_loss(loss)
        self.test_ppl(torch.exp(loss))
        self.test_bleu(bleu_score)

        # log
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/ppl", self.test_ppl, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/bleu", self.test_bleu, on_step=False, on_epoch=True, prog_bar=True)

        return loss
        

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}
        
        