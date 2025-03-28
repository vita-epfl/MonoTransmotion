from process_args import get_eval_args
from eval.evaluator import Evaluator


def main():
    args = get_eval_args()
    eval_mode = args.eval_mode
    assert eval_mode in ["loc", "traj_pred"]
    evaluator = Evaluator(args)
    if eval_mode == "loc":
        evaluator.evaluate_loc()
    elif eval_mode == "traj_pred":
        evaluator.evaluate_traj_pred()


if __name__ == '__main__':
    main()