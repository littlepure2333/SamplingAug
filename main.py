import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        utility.print_params(_model,checkpoint,args)
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

    checkpoint.done()

if __name__ == '__main__':
    main()
