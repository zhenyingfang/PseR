import os
import numpy as np
import options
from dataset import dataset
import torch.utils.data as data


if __name__ == '__main__':
    args = options.parse_args()

    train_loader = data.DataLoader(dataset(args, phase="train", sample="random", stage=args.stage), 
                                batch_size=1, shuffle=True, num_workers=args.num_workers)
    
