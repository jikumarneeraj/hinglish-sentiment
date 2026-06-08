# paper/

Place your research paper draft here.

## Recommended Templates

### IEEE Format
Download the IEEE conference template:
- https://www.ieee.org/conferences/publishing/templates.html
- Use: `IEEE_conference_template.docx` or `conference_101719.zip` (LaTeX)

### Springer Format
- https://www.springer.com/gp/authors-editors/conference-proceedings/conference-proceedings-guidelines

---

## Paper Structure Checklist

- [ ] **Abstract** (150–250 words): problem, method, key result
- [ ] **Introduction**: why Hinglish NLP is hard, your 3 contributions
- [ ] **Related Work**: 8–10 summarised papers on Hinglish NLP + sentiment analysis
- [ ] **Dataset**: SemEval 2020 Task 9 description + EDA figures
- [ ] **Methodology**: all 4 models described with equations / architecture diagrams
- [ ] **Experiments**: preprocessing steps, hyperparameters, hardware, metrics
- [ ] **Results**: Table 1 (from results_table.csv) + model_comparison_chart.png + confusion matrices
- [ ] **Error Analysis**: 10–15 misclassified examples explained
- [ ] **Conclusion**: findings, limitations, future work

---

## Figures to Include (from results/)

| Figure file | Section | Caption |
|-------------|---------|---------|
| `class_distribution.png` | Dataset | Class distribution of SemEval 2020 Task 9 |
| `wordclouds.png` | Dataset | Word clouds by sentiment class |
| `tweet_length_distribution.png` | Dataset | Tweet length distribution |
| `model_comparison_chart.png` | Results | Performance comparison across all models |
| `all_confusion_matrices.png` | Results | Confusion matrices for all models |
| `lstm_training_curves.png` | Experiments | LSTM training and validation curves |

---

## Target Venues

| Venue | Deadline (check annually) | Format |
|-------|--------------------------|--------|
| IEEE INDICON | ~August | IEEE |
| ICCES | ~varies | IEEE |
| Elsevier Expert Systems with Applications | Rolling | Elsevier |

---

*Cite every dataset, pretrained model, and library you use — including the MuRIL paper and SemEval dataset paper.*
