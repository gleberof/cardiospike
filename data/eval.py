import csv
import os
from typing import Dict, Generator, List


def read_csv(path: str) -> Generator[Dict[str, str], None, None]:
    with open(path, "r") as f:
        for line in csv.DictReader(f):
            yield line


def confusion_matrix(
    true_path: str, pred_path: str, key_columns: List[str], value_column: str
) -> List[List[int]]:
    assert os.path.isfile(true_path), f"{true_path} not found"
    assert os.path.isfile(pred_path), f"{pred_path} not found"

    counts = [[0, 0], [0, 0]]  # [[tn, fp], [fn, tp]]

    for line_num, (true_row, pred_row) in enumerate(
        zip(read_csv(true_path), read_csv(pred_path))
    ):
        for col in key_columns + [value_column]:
            assert col in true_row, f"no {col} in ground truth (line {line_num})"
            assert col in pred_row, f"no {col} in prediction (line {line_num})"

        y_true = true_row[value_column]
        y_pred = pred_row[value_column]

        assert y_true in {"0", "1"}, f"invalid y_true = {y_true} (line {line_num})"
        assert y_pred in {"0", "1"}, f"invalid y_ppred = {y_pred} (line {line_num})"

        counts[int(y_true)][int(y_pred)] += 1
    return counts


def f1_score(confmat: List[List[int]]) -> float:
    (tn, fp), (fn, tp) = confmat
    return tp / (tp + 1 / 2 * (fp + fn))


if __name__ == "__main__":
    import argparse

    MAX_SIZE = 1e6

    parser = argparse.ArgumentParser(
        description="Computes F1 score. For example on file format check `sample_submission.csv`"
    )
    parser.add_argument("pred_path", help="path to .csv file with predictions")
    parser.add_argument("true_path", help="path to .csv file with ground truth")

    args = parser.parse_args()

    assert os.path.getsize(args.true_path) < MAX_SIZE, f"{args.true_path} too big"
    assert os.path.getsize(args.pred_path) < MAX_SIZE, f"{args.pred_path} too big"

    confmat = confusion_matrix(
        true_path=args.true_path,
        pred_path=args.pred_path,
        key_columns=["id", "time"],
        value_column="y",
    )

    score = f1_score(confmat)
    print(score)
