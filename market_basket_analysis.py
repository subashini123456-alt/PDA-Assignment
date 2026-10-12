from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt

transactions = [
    ["tea", "sugar", "biscuits"],
    ["tea", "sugar"],
    ["coffee", "sugar", "biscuits"],
    ["tea", "biscuits"],
    ["coffee", "sugar"],
    ["tea", "sugar", "biscuits", "bread"],
    ["bread", "biscuits"],
    ["tea", "sugar", "bread"],
    ["coffee", "biscuits"],
    ["tea", "sugar", "biscuits"],
]
n = len(transactions)
min_support, min_conf = 0.3, 0.7

def support(itemset):
    return sum(set(itemset) <= set(t) for t in transactions) / n

# ---- Apriori: find frequent itemsets level by level ----
items = sorted({i for t in transactions for i in t})
current = [frozenset([i]) for i in items]
frequent, k = {}, 1
while current:
    level = {s: support(s) for s in current if support(s) >= min_support}
    frequent.update(level)
    keys = list(level)
    current = list({a | b for a in keys for b in keys if len(a | b) == k + 1})
    k += 1

fi = pd.DataFrame([(", ".join(sorted(s)), v) for s, v in frequent.items()],
                  columns=["itemset", "support"])
fi = fi.sort_values("support", ascending=False).reset_index(drop=True)
print("Frequent Itemsets (min_support = 0.3):")
print(fi.to_string())

# ---- Association rules ----
rows = []
for s in frequent:
    if len(s) < 2:
        continue
    for r in range(1, len(s)):
        for A in combinations(sorted(s), r):
            A = frozenset(A)
            B = s - A
            conf = frequent[s] / support(A)
            if conf >= min_conf:
                rows.append((", ".join(sorted(A)), ", ".join(sorted(B)),
                             frequent[s], conf, conf / support(B)))
rules = pd.DataFrame(rows, columns=["antecedent", "consequent",
                                    "support", "confidence", "lift"])
rules = rules.sort_values("lift", ascending=False).round(3).reset_index(drop=True)
print("\nAssociation Rules (min_confidence = 0.7):")
print(rules.to_string())

# ---- Charts ----
fi.plot.barh(x="itemset", y="support", legend=False, color="teal")
plt.gca().invert_yaxis()
plt.xlabel("Support")
plt.title("Frequent Itemsets")
plt.tight_layout()
plt.show()

plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 150, alpha=0.6, color="crimson")
for _, r in rules.iterrows():
    plt.annotate(r["antecedent"] + " -> " + r["consequent"],
                 (r["support"], r["confidence"]), fontsize=7)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Association Rules (bubble size = lift)")
plt.show()
