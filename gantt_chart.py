import matplotlib.pyplot as plt
from datetime import datetime, timedelta

start = datetime(2025, 11, 1)
nov_end = datetime(2025, 11, 30)

tasks_nov = [
    ("Topic + Literature", start, start + timedelta(days=10)),
    ("Data Cleaning", start + timedelta(days=8), start + timedelta(days=15)),
    ("TF-IDF Baseline", start + timedelta(days=14), start + timedelta(days=20)),
    ("BERT Quick-Test", start + timedelta(days=20), nov_end),
]

future = [
    ("BERT Full Grid", datetime(2025, 12, 1), datetime(2026, 1, 15)),
    ("TextCNN + HAN", datetime(2026, 1, 16), datetime(2026, 2, 28)),
    ("Compression + Quant", datetime(2026, 3, 1), datetime(2026, 4, 15)),
    ("Demo + Thesis", datetime(2026, 4, 16), datetime(2026, 5, 15)),
    ("Defence", datetime(2026, 5, 16), datetime(2026, 5, 31)),
]

fig, ax = plt.subplots(figsize=(12, 4))
y = 0
for name, s, e in tasks_nov:
    ax.barh(y, (e - s).days, left=(s - start).days, color='#4CAF50', label='Completed' if y == 0 else "")
    ax.text((s + (e - s) / 2 - start).days, y, name, ha='center', va='center', fontsize=9, color='white')
    y += 1

for name, s, e in future:
    ax.barh(y, (e - s).days, left=(s - start).days, color='#E0E0E0', label='Future' if y == len(tasks_nov) else "")
    ax.text((s + (e - s) / 2 - start).days, y, name, ha='center', va='center', fontsize=9, color='black')
    y += 1

ax.set_xlabel("Nov 2025 → May 2026 (Days)")
ax.set_ylabel("Task")
ax.set_title("Gantt Chart – Updated to Nov 30, 2025")
ax.set_xticks(range(0, 211, 30))
ax.set_xticklabels([(start + timedelta(days=d)).strftime('%Y-%m-%d') for d in range(0, 211, 30)], rotation=45)
ax.grid(axis='x', alpha=0.3)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("figs/gantt_nov_en.png", dpi=300)
plt.close()
print("✅ English Gantt saved (updated to Nov 30, 2025)")