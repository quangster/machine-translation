import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric, MaxMetric
from typing import Tuple
import torch.nn as nn
from transformers import get_scheduler


class TransformerModuleV2(LightningModule):
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
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        """ Initialize a `TransformerModuleV2`.

        Args:
            net (torch.nn.Module): The model to train
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use for training.
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.criterion = criterion
        self._reset_parameters()

        # metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_ppl = MeanMetric()
        self.val_ppl = MeanMetric()
        self.test_ppl = MeanMetric()
        
        self.val_bleu = MaxMetric()
        self.val_loss_best = MinMetric()
        self.val_bleu_best = MaxMetric()
        self.test_bleu = MaxMetric()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_masks(self, src, trg, src_pad, trg_pad):
        """
        Tạo mask cho encoder và decoder:
        - src_mask: Che padding token trong chuỗi nguồn.
        - trg_mask: Kết hợp mask padding và look-ahead mask trong chuỗi đích.
        """
        # Tạo src_mask: (batch_size, 1, seq_length_src)
        src_mask = (src != src_pad).unsqueeze(1)

        device = src_mask.device

        trg_mask = None
        if trg is not None:
            # Mask padding: (batch_size, 1, seq_length_trg)
            trg_padding_mask = (trg != trg_pad).unsqueeze(1)

            # Mask look-ahead: (seq_length_trg, seq_length_trg)
            trg_seq_len = trg.size(1)
            look_ahead_mask = torch.triu(torch.ones((trg_seq_len, trg_seq_len), device=device), diagonal=1).bool()

            # Kết hợp padding mask và look-ahead mask: (batch_size, seq_length_trg, seq_length_trg)
            trg_mask = trg_padding_mask & ~look_ahead_mask.unsqueeze(0)
        return src_mask, trg_mask

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.net(src, tgt, src_mask, tgt_mask)
    
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

        src_mask, trg_mask = self._create_masks(src, tgt_input, 0, 0)
        logits = self.net(src, tgt_input, src_mask, trg_mask)
        loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_out.contiguous().view(-1))
        return loss

    def training_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        
        self.net.train()
        loss = self.model_step(batch, batch_idx)

        # update metrics
        self.train_loss(loss)
        self.train_ppl(torch.exp(loss))

        # log
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ppl", self.train_ppl, on_step=True, on_epoch=True, prog_bar=True)

        # # log learning rate
        # self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        
        self.net.eval()
        with torch.inference_mode():
            loss = self.model_step(batch, batch_idx)

        # update metrics
        self.val_loss(loss)
        self.val_ppl(torch.exp(loss))

        # log
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/ppl", self.val_ppl, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute() # get current val loss
        self.val_loss_best(loss) # update best so far val loss

        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        
        self.net.eval()
        with torch.inference_mode():
            loss = self.model_step(batch, batch_idx)

        # update metrics
        self.test_loss(loss)
        self.test_ppl(torch.exp(loss))

        # log
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/ppl", self.test_ppl, on_step=True, on_epoch=True, prog_bar=True)

        return loss
        
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        lr_scheduler = get_scheduler(
            "linear", optimizer, num_warmup_steps=4000, 
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
        
        