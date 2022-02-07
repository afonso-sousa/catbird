from pathlib import Path

from catbird.core import Config, build_optimizer
from torch import optim
import torch
import math

data_path = Path(__file__).parent / "data"


def test_opt():
    cfg_file = Path(data_path, "config/dummy_optim_cfg.py")
    cfg = Config.fromfile(cfg_file)
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)
    
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1)
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')

    optimizer = build_optimizer(model, cfg.optimizer)
    assert isinstance(optimizer, optim.Optimizer)
    assert isinstance(optimizer, optim.SGD)
    y_pred = model(xx)
    
    loss = loss_fn(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
