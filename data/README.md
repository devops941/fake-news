# Dataset

`Fake.csv` and `True.csv` in this folder are the real **"Fake and Real News Dataset"**
by Clement Bisaillon (23,481 fake + 21,417 real = 44,898 articles, 2016-2017 US political
news), each with columns `title, text, subject, date`. Obtained via a verified public
mirror on Hugging Face (`haoxianc/gilchr_fake-and-real-news-dataset`) since the original
Kaggle page requires an account login — row counts, file sizes, column schema, subject
categories, and the Reuters-dateline pattern in real articles all match the original
dataset exactly.

Original source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
(license: CC BY-NC-SA 4.0 — non-commercial, educational/research use, which this
capstone project falls under).

If you ever need to re-fetch it:

1. **Manual**: download from the Kaggle page above (requires a free account), extract
   `Fake.csv` and `True.csv` here.
2. **Kaggle API**: `pip install kaggle`, place your `kaggle.json` token in
   `~/.kaggle/kaggle.json`, then:
   ```
   kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data --unzip
   ```
3. **HF mirror** (no login needed):
   ```
   curl -L -o data/_combined.csv "https://huggingface.co/datasets/haoxianc/gilchr_fake-and-real-news-dataset/resolve/main/fake_real_news.csv"
   ```
   then split by its `target` column (1 = fake, 0 = real) into `Fake.csv` / `True.csv`
   with columns `title, text, subject, date` (drop `target`).

## Synthetic sample (no longer used, kept for reference)

`scripts/generate_sample_data.py` generates a tiny 300-row synthetic stand-in dataset,
useful only for quickly smoke-testing the pipeline without the real 44,898-row data.
Re-running it will overwrite `Fake.csv`/`True.csv` with synthetic data — don't run it
unless you intend to replace the real dataset.
