

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import json
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from solo.args.corr import parse_args_corr
from solo.data.classification_dataloader import prepare_data
from solo.methods import METHODS
from solo.utils.corr import calculate_correlation


def main():
    args = parse_args_corr()

    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = Path(args.pretrained_checkpoint)

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)
    cfg = OmegaConf.create(method_args)

    # build the model
    model = (
        METHODS[method_args["method"]]
        .load_from_checkpoint(ckpt_path, strict=False, cfg=cfg)
        .backbone
    )

    # prepare data
    train_loader, val_loader = prepare_data(
        args.dataset,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        data_format=args.data_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        auto_augment=False,
    )

    # move model to the gpu
    device = "cuda:0"
    model = model.to(device)
    
    print(cfg)
    name = cfg['name']
    wandb_run_id = cfg['wandb_run_id']
    print("CFG: ", name, wandb_run_id)

    calculate_correlation(device, model, train_loader,
                          f'./correlation_analysis/{name}_{wandb_run_id}_train')
    calculate_correlation(device, model, val_loader,
                          f'./correlation_analysis/{name}_{wandb_run_id}_val')


if __name__ == "__main__":
    main()
