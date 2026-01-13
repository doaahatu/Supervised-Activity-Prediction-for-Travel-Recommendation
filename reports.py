import csv
import os
from collections import defaultdict, Counter

DATASET_FILE = "ALL_DATA_2_NORMALIZED.csv"
BASE_REPORT_DIR = "reports"

UNIQUE_DIR = os.path.join(BASE_REPORT_DIR, "unique_values")
COUNT_DIR = os.path.join(BASE_REPORT_DIR, "value_counts")

os.makedirs(UNIQUE_DIR, exist_ok=True)
os.makedirs(COUNT_DIR, exist_ok=True)

rows = []
missing_counts = Counter()

with open(DATASET_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    for row in reader:
        rows.append(row)
        for field in fieldnames:
            if not row[field] or not row[field].strip():
                missing_counts[field] += 1

total_rows = len(rows)

# ---------- 1. Unique values per field WITH COUNTS ----------
unique_values = defaultdict(set)
value_counts = defaultdict(Counter)

for row in rows:
    for field in fieldnames:
        value = row[field].strip()
        if value:
            unique_values[field].add(value)
            value_counts[field][value] += 1

# Write unique values WITH counts
for field, values in unique_values.items():
    path = os.path.join(UNIQUE_DIR, f"{field}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Unique values for '{field}' ({len(values)} total)\n")
        f.write("=" * 70 + "\n\n")

        # Sort by count (descending), then by value name
        sorted_values = sorted(values, key=lambda v: (-value_counts[field][v], v))

        for v in sorted_values:
            count = value_counts[field][v]
            percentage = (count / total_rows) * 100
            f.write(f"{v:<40} {count:>6} ({percentage:>5.2f}%)\n")

# ---------- 2. Value frequency reports (CSV format) ----------
for field, counter in value_counts.items():
    path = os.path.join(COUNT_DIR, f"{field}_counts.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([field, "count", "percentage"])
        for value, count in counter.most_common():
            percentage = (count / total_rows) * 100
            writer.writerow([value, count, f"{percentage:.2f}%"])

# ---------- 3. Missing values report ----------
with open(os.path.join(BASE_REPORT_DIR, "missing_values_report.txt"), "w", encoding="utf-8") as f:
    f.write("Missing Values Report\n")
    f.write("=" * 40 + "\n\n")
    for field in fieldnames:
        count = missing_counts[field]
        percent = (count / total_rows) * 100
        f.write(f"{field:<20} {count:>6} missing ({percent:>5.2f}%)\n")

# ---------- 4. Dataset summary ----------
with open(os.path.join(BASE_REPORT_DIR, "dataset_summary.txt"), "w", encoding="utf-8") as f:
    f.write("Dataset Summary Report\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Total rows: {total_rows}\n")
    f.write(f"Total columns: {len(fieldnames)}\n\n")

    f.write("Columns:\n")
    for field in fieldnames:
        f.write(f"  - {field:<20} ({len(unique_values[field]):>3} unique values)\n")

    f.write("\n" + "=" * 40 + "\n")
    f.write("Top 5 values per field:\n")
    f.write("=" * 40 + "\n\n")

    for field in fieldnames:
        f.write(f"{field}:\n")
        for value, count in value_counts[field].most_common(5):
            percentage = (count / total_rows) * 100
            f.write(f"  {value:<30} {count:>6} ({percentage:>5.2f}%)\n")
        f.write("\n")

print("✅ Validation complete. Reports saved in 'reports/' folder.")
print(f"\nGenerated reports:")
print(f"  📁 {UNIQUE_DIR}/ - Unique values with counts")
print(f"  📁 {COUNT_DIR}/ - Value frequency CSVs")
print(f"  📄 {os.path.join(BASE_REPORT_DIR, 'missing_values_report.txt')}")
print(f"  📄 {os.path.join(BASE_REPORT_DIR, 'dataset_summary.txt')}")