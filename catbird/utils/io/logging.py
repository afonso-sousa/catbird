"""File with logging facilities."""
from logging import Logger
from typing import Any, Dict, Optional


def log_metrics(
    logger: Logger,
    elapsed: float,
    tag: str,
    metrics: Dict[str, Any],
    epoch: Optional[int] = None,
) -> None:
    """Log metric evaluation score.

    Args:
        logger (Logger): Logger instance.
        elapsed (float): Time that evaluation took.
        tag (str): An identification tag.
        metrics (Dict[str, Any]): Dictionary with evaluation metrics' outputs.
        epoch (Optional[int]): Epoch number.
    """
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    epoch_string = f"Epoch: {epoch} - " if epoch else ""
    logger.info(
        f"\n{epoch_string}Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


# def log_basic_info(logger: Logger, config: Config) -> None:
#     """Log environment and config information.

#     Args:
#         logger (Logger): Logger instance.
#         config (Config): Configuration information.
#     """
#     logger.info(f"Train on {config.data.dataset_name}")
#     logger.info(f"- PyTorch version: {torch.__version__}")
#     logger.info(f"- Ignite version: {ignite.__version__}")
#     if torch.cuda.is_available():
#         # explicitly import cudnn as torch.backends.cudnn can not be pickled with hvd spawning procs
#         from torch.backends import cudnn

#         logger.info(
#             f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}"
#         )
#         logger.info(f"- CUDA version: {torch.version.cuda}")
#         logger.info(f"- CUDNN version: {cudnn.version()}")

#     logger.info("\n")
#     logger.info("Configuration:")
#     for key, value in config.items():
#         logger.info(f"\t{key}: {value}")
#     logger.info("\n")

#     if idist.get_world_size() > 1:
#         logger.info("\nDistributed setting:")
#         logger.info(f"\tbackend: {idist.backend()}")
#         logger.info(f"\tworld size: {idist.get_world_size()}")
#         logger.info("\n")
