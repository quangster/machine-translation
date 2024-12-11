from omegaconf import DictConfig
import rootutils
import hydra
import logging

from lightning.pytorch.callbacks import RichModelSummary, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning import LightningDataModule, LightningModule, Trainer
import lightning as L


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True) 

from src.utils.log_utils import log_hyperparameters, task_wrapper  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

@task_wrapper
def train(cfg: DictConfig):
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{cfg.paths.output_dir}/checkpoints",
            filename="epoch-{epoch:03d}",
            monitor="val/loss",
            mode="min",
            save_last=True,
            auto_insert_metric_name=False,
        ),
        RichProgressBar(),
        RichModelSummary(max_depth=1)
    ]
    

    log.info("Instantiating loggers...")
    logger: WandbLogger = hydra.utils.instantiate(cfg.logger.wandb) if cfg.get("logger").get("wandb") else None

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
    
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    log.info(f"Best ckpt path: {trainer.checkpoint_callback.best_model_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    metric_dict, _ = train(cfg)
    return None

if __name__ == "__main__":
    main()