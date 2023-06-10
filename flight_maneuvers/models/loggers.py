import types
import inspect
import pathlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Callable, Optional, Union, Sequence
import torch.distributed as dist


class Logger(ABC):
    @abstractmethod
    def log_hyperparams(self, params):
        pass

    @abstractmethod
    def log_metrics(self, metrics, step=None):
        pass

    @staticmethod
    def _sanitize_params(params):
        def _sanitize(val):
            if isinstance(val, Callable):
                try:
                    _val = val()
                    if isinstance(_val, Callable):
                        return val.__name__
                    return _val
                except Exception:
                    return getattr(val, "__name__", None)
            elif isinstance(val, pathlib.Path) or isinstance(val, Enum):
                return str(val)
            return val

        return {key: _sanitize(val) for key, val in params.items()}

def save_hyperparameters(frame):
    # the frame needs to be created in this file.
    logger = frame.f_locals['self']
    gym = frame.f_back.f_locals['self']
    logger.model = gym.model
    logger.hparams = gym.hparams

try:
    from torch.utils.tensorboard import SummaryWriter
    class TBLogger(SummaryWriter):
        def __init__(self, save_dir):
            super().__init__(str(save_dir))
            save_hyperparameters(inspect.currentframe())

        def log_stochastic(self):
            self.add_scalars()

        def log_metrics(self, metrics, step=None):
            if step is None:
                step = 0
            self.add_scalar_dict(metrics, step)
except ImportError:
    pass    


class LoggerCollection(Logger):
    def __init__(self, loggers):
        super().__init__()
        self.loggers = loggers

    def __getitem__(self, index):
        return [logger for logger in self.loggers][index]

    def log_metrics(self, metrics, step=None):
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def log_hyperparams(self, params):
        for logger in self.loggers:
            logger.log_hyperparams(params)


try:
    import dllogger
    from dllogger import Verbosity
    class DLLogger(Logger):
        def __init__(self, save_dir: pathlib.Path, filename: str):
            super().__init__()
            if not dist.is_initialized() or dist.get_rank() == 0:
                save_dir.mkdir(parents=True, exist_ok=True)
                dllogger.init(
                    backends=[dllogger.JSONStreamBackend(Verbosity.DEFAULT, str(save_dir / filename))])

        def log_hyperparams(self, params):
            params = self._sanitize_params(params)
            dllogger.log(step="PARAMETER", data=params)

        def log_metrics(self, metrics, step=None):
            if step is None:
                step = tuple()

            dllogger.log(step=step, data=metrics)
except ImportError:
    pass


try:
    import wandb
    class WandbLogger(Logger):
        def __init__(
                self,
                name: str,
                save_dir: pathlib.Path,
                id: Optional[str] = None,
                project: Optional[str] = None
        ):
            super().__init__()
            if not dist.is_initialized() or dist.get_rank() == 0:
                save_dir.mkdir(parents=True, exist_ok=True)
                self.experiment = wandb.init(name=name,
                                            project=project,
                                            id=id,
                                            dir=str(save_dir),
                                            resume='allow',
                                            anonymous='must')

        def log_hyperparams(self, params: Dict[str, Any]) -> None:
            params = self._sanitize_params(params)
            self.experiment.config.update(params, allow_val_change=True)

        def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
            if step is not None:
                self.experiment.log({**metrics, 'epoch': step})
            else:
                self.experiment.log(metrics)
except ImportError:
    pass
