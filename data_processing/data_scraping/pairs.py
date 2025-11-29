import time
import re
import pandas as pd

def clean_line(line):
    return re.sub(r"^\[\d+\]\s*", "", line).strip()


spanish = {
    "MAT": list(range(1, 29)),
    "MRK": list(range(1, 17)),
    "LUK": list(range(1, 25)),
    "JHN": list(range(1, 22)),
    "ACT": list(range(1, 29)),
    "ROM": list(range(1, 17)),
    "1CO": list(range(1, 17)),
    "2CO": list(range(1, 14)),
    "GAL": list(range(1, 7)),
    "EPH": list(range(1, 7)),
    "PHP": list(range(1, 5)),
    "COL": list(range(1, 5)),
    "1TH": list(range(1, 6)),
    "2TH": list(range(1, 4)),
    "1TI": list(range(1, 7)),
    "2TI": list(range(1, 5)),
    "TIT": list(range(1, 4)),
    "PHM": list(range(1, 2)),
    "HEB": list(range(1, 14)),
    "JAS": list(range(1, 6)),
    "1PE": list(range(1, 6)),
    "2PE": list(range(1, 4)),
    "1JN": list(range(1, 6)),
    "2JN": list(range(1, 2)),
    "3JN": list(range(1, 2)),
    "JUD": list(range(1, 2)),
    "REV": list(range(1, 23)),
}

spanish_clean = []
warao_clean = []

for book, chapters in spanish.items():
    for chapter in chapters:
        spanish_filename = f"spanish_{book}_{chapter}.txt"
        warao_filename = f"warao_{book}_{chapter}.txt"
        # Read the two text files
        with open(warao_filename, "r", encoding="utf-8") as f:
            warao_sentences = f.read().splitlines()

        with open(spanish_filename, "r", encoding="utf-8") as f:
            spanish_sentences = f.read().splitlines()
        # print(len(warao_sentences))
        # print(len(spanish_sentences))

        for i in range(len(spanish_sentences)):
            warao_sentence = clean_line(warao_sentences[i])
            warao_clean.append(warao_sentence)
            spanish_sentence = clean_line(spanish_sentences[i])
            spanish_clean.append(spanish_sentence)
            # print(spanish_sentence)
            # print(warao_sentence)
            # time.sleep(5)

# Create a DataFrame of pairs
df = pd.DataFrame({
    "warao": warao_clean,
    "spanish": spanish_clean
})

# Save to CSV
df.to_csv("Bible_verse_pairs.csv", index=False, encoding="utf-8")
