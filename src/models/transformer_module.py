import torch
from lightning import LightningModule
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
        self.train_ppl = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_ppl = MeanMetric()
        self.val_bleu = MaxMetric()

        self.val_loss_best = MinMetric()
        self.val_bleu_best = MaxMetric()

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
        # memory_key_padding_mask: (batch_size, src_seq_len
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

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.net.create_mask(src, tgt_input, self.device)
        
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
        return loss

    def training_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        loss = self.model_step(batch, batch_idx)

        # update metrics
        self.train_loss(loss)
        self.train_ppl(torch.exp(loss))

        # log
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ppl", self.train_ppl, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        loss = self.model_step(batch, batch_idx)

        # update metrics
        self.val_loss(loss)
        self.val_ppl(torch.exp(loss))

        # log
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ppl", self.val_ppl, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute() # get current val loss
        self.val_loss_best(loss) # update best so far val loss

        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}
        
        