# data/raw/

Place the SemEval 2020 Task 9 Hinglish Sentiment dataset here.

## Download Instructions

1. Visit: https://ritual.uh.edu/semeval-2020-task9/
2. Download the Hinglish (Hindi-English) training file
3. Rename or place the CSV file as: `hinglish_train.csv`

Expected file path: `data/raw/hinglish_train.csv`

## Expected Format

The CSV should contain at least these two columns:

| Column | Values |
|--------|--------|
| `text` (or `tweet`) | raw tweet text |
| `label` (or `sentiment`) | `positive`, `negative`, or `neutral` |

`preprocess.py` automatically handles common column name variations.

## Citation

```
@inproceedings{patwa2020semeval,
  title={SemEval-2020 Task 9: Sentiment Analysis for Code-Mixed Social Media Text},
  author={Patwa, Parth and Aguilar, Gustavo and Kar, Sudipta and others},
  booktitle={Proceedings of the 14th International Workshop on Semantic Evaluation (SemEval-2020)},
  pages={624--634},
  year={2020}
}
```
