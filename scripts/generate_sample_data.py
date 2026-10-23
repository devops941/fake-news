"""Generate a small SYNTHETIC Fake.csv / True.csv for development and demos.

This is NOT the real Kaggle dataset. It exists so the training pipeline and the
Flask app can be exercised end-to-end before the real ~44,000-row dataset is
downloaded. See data/README.md for how to swap in the real data.

Run from the project root:
    python scripts/generate_sample_data.py
"""

import csv
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CITIES = [
    "WASHINGTON", "NEW YORK", "LONDON", "PARIS", "BERLIN", "TOKYO",
    "BEIJING", "MOSCOW", "DELHI", "SYDNEY", "TORONTO", "CHICAGO",
]
ENTITIES = [
    "the Federal Reserve", "the United Nations", "the European Commission",
    "the Supreme Court", "the Senate", "the World Health Organization",
    "the central bank", "the prime minister's office", "city officials",
    "the trade ministry", "the health department", "the national election commission",
]
REAL_ACTIONS = [
    "announced a new policy aimed at reducing inflation over the next fiscal year",
    "said it would review trade agreements with neighboring countries",
    "released a report on employment figures for the past quarter",
    "confirmed that talks with opposition leaders would continue next week",
    "approved funding for a new infrastructure project",
    "issued a statement clarifying its position on recent tariff changes",
    "held a press conference to discuss the latest economic indicators",
    "signed an agreement to expand cooperation on climate initiatives",
    "postponed a scheduled vote after further committee review",
    "published updated guidance following the latest public health data",
]
REAL_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
REAL_SUBJECTS = ["politicsNews", "worldnews", "Government News", "economy"]

FAKE_HOOKS = [
    "You won't believe what {entity} just did",
    "SHOCKING: {entity} caught hiding the truth from the public",
    "Leaked documents reveal {entity} secretly plans to control your money",
    "Scientists refuse to tell you this one simple fact about {entity}",
    "BREAKING: {entity} exposed in massive cover-up nobody is talking about",
    "This video proves {entity} has been lying to everyone for years",
    "Insider reveals the terrifying secret plan behind {entity}",
    "Mainstream media won't report this bombshell about {entity}",
]
FAKE_BODY = [
    "Share this before it gets deleted! Sources say the real agenda is being hidden from ordinary citizens.",
    "According to an anonymous insider, the plan has been in motion for years and nobody wants you to know.",
    "Wake up! This is bigger than anyone imagined and the evidence is being scrubbed from the internet.",
    "Forward this to everyone you know before the truth disappears forever.",
    "What they don't want you to see will shock you to your core, according to leaked chat logs.",
    "A whistleblower claims the cover-up goes all the way to the top, but no major outlet will touch it.",
    "The elites are terrified this information is finally getting out, insiders say.",
    "This changes everything we thought we knew, and the silence from officials speaks volumes.",
]
FAKE_SUBJECTS = ["News", "politics", "conspiracy", "left-news"]


def make_real_row():
    city = random.choice(CITIES)
    entity = random.choice(ENTITIES)
    action = random.choice(REAL_ACTIONS)
    day = random.choice(REAL_DAYS)
    title = f"{entity.title()} {action[:40].rstrip()}..."
    text = f"{city} (Reuters) - {entity.capitalize()} {action} on {day}. Officials said further details would be provided in the coming weeks."
    subject = random.choice(REAL_SUBJECTS)
    date = f"{random.randint(1, 28)}-{random.choice(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}-{random.choice(['2016','2017'])}"
    return [title, text, subject, date]


def make_fake_row():
    entity = random.choice(ENTITIES)
    hook = random.choice(FAKE_HOOKS).format(entity=entity)
    body = random.choice(FAKE_BODY)
    title = hook
    text = f"{hook}! {body} {random.choice(FAKE_BODY)}"
    subject = random.choice(FAKE_SUBJECTS)
    date = f"{random.choice(['January','February','March','April','May','June'])} {random.randint(1, 28)}, {random.choice(['2016','2017'])}"
    return [title, text, subject, date]


def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "text", "subject", "date"])
        writer.writerows(rows)


def main(n_per_class: int = 150):
    DATA_DIR.mkdir(exist_ok=True)
    real_rows = [make_real_row() for _ in range(n_per_class)]
    fake_rows = [make_fake_row() for _ in range(n_per_class)]
    write_csv(DATA_DIR / "True.csv", real_rows)
    write_csv(DATA_DIR / "Fake.csv", fake_rows)
    print(f"Wrote {len(real_rows)} rows to {DATA_DIR / 'True.csv'}")
    print(f"Wrote {len(fake_rows)} rows to {DATA_DIR / 'Fake.csv'}")


if __name__ == "__main__":
    main()
