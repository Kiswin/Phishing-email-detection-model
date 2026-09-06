# Phishing Email Detection Model

A scikit-learn pipeline that classifies emails as **Phishing** or **Safe**.

## Files
- `generate_dataset.py` – builds a labeled synthetic dataset (`emails_dataset.csv`) of phishing and legitimate emails, with randomized names, URLs, and phrasing so the model can't just memorize fixed templates.
- `phishing_detector.py` – extracts features, trains the model, and evaluates it.
- `emails_dataset.csv` – the generated dataset (700 emails, ~50/50 split).
- `confusion_matrix.png` – the confusion matrix produced by the run.

## How it works
1. **Text features:** TF-IDF (unigrams + bigrams) on the raw email text.
2. **Engineered features:** number of URLs, IP-based URLs (a classic phishing red flag), HTTPS presence, exclamation marks, count of suspicious keywords ("urgent", "verify", "suspend", etc.), text length, digit count, dollar-sign presence.
3. **Model:** Random Forest Classifier (300 trees) trained on the combined feature set.
4. **Evaluation:** accuracy, precision/recall/F1 per class, and a confusion matrix plot.

## Run it yourself
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python generate_dataset.py      # creates emails_dataset.csv
python phishing_detector.py     # trains model, prints metrics, saves confusion_matrix.png
```

## Honest note on the dataset
This dataset is **synthetically generated** (not a real-world email corpus), since none was provided with the assignment. That means:
- The accuracy numbers (near-100%) are inflated compared to what you'd see on a real dataset like Nazario's phishing corpus or the Kaggle "Phishing Email Detection" dataset, because the synthetic phishing/legit emails are more clearly separated than messy real-world mail.
- I added randomized names, URLs, and wording specifically so the model learns actual patterns (urgency language, suspicious URLs) rather than memorizing exact sentences — verified this by testing it on brand-new example emails it had never seen (see the bottom of `phishing_detector.py`), and it classified them correctly.
- **For a stronger submission**, swap `emails_dataset.csv` for a real public dataset (just keep the `text`/`label` column format) — the rest of the pipeline will work unchanged.

## Possible extensions
- Try `LogisticRegression` or `MultinomialNB` and compare against the Random Forest.
- Add cross-validation instead of a single train/test split.
- Add more engineered features: sender domain mismatch, presence of attachments, spelling error rate.
