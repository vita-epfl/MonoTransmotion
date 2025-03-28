from process_args import get_train_args
from train.trainer import Trainer

def main():
    args = get_train_args()
    train_mode = args.train_mode
    assert train_mode in ["joint", "freeze_loc", "loc"] 
    trainer = Trainer(args)
    if train_mode == "joint":
        trainer.train_joint()
    elif train_mode == "freeze_loc":
        trainer.train_freeze_loc()
    elif train_mode == "loc":
        trainer.train_loc()      


if __name__ == '__main__':
    main()