"""
IECSAP - Concept 6: Market Basket Analysis (Week 15)
Mines co-purchase patterns from order history to power a "customers who
bought X also bought Y" recommender -- surfaced by the chatbot after it
resolves a support ticket ("since your Wireless Mouse issue is sorted,
customers who bought that also bought a Mechanical Keyboard").

Approach: Apriori algorithm (mlxtend) over one-hot encoded baskets, producing
frequent itemsets and association rules ranked by support / confidence / lift.
"""
from pathlib import Path
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

BASE = Path(__file__).resolve().parent.parent


def load_baskets():
    df = pd.read_csv(BASE / "data" / "transactions.csv")
    return df.groupby("basket_id")["item"].apply(list).tolist()


def mine_rules(min_support=0.015, min_confidence=0.2):
    baskets = load_baskets()
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    onehot = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(onehot, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values("lift", ascending=False)
    return frequent_itemsets, rules


def recommend(product: str, rules: pd.DataFrame, top_k=3):
    """Given a single product the customer already has, recommend cross-sell
    items using rules where {product} is in the antecedent. Used as a
    fallback when we only know a product the customer MENTIONED, not their
    real order contents."""
    hits = rules[rules["antecedents"].apply(lambda s: product in s and len(s) == 1)]
    hits = hits.sort_values("lift", ascending=False).head(top_k)
    consequent_strs = hits["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return list(zip(consequent_strs, hits["confidence"].round(2), hits["lift"].round(2)))


def recommend_for_items(items, rules: pd.DataFrame, top_k=3):
    """Given ALL the items in a customer's actual order (looked up via
    OrderStore, not just one product mentioned in text), recommend
    cross-sell items: the union of single-item rules triggered by anything
    already in the order, deduplicated, ranked by lift, excluding items the
    customer already has. This is the most personalized recommendation
    tier -- it reflects the real order, not a guess from free text."""
    owned = set(items)
    best = {}  # consequent -> (confidence, lift)
    for item in items:
        hits = rules[rules["antecedents"].apply(lambda s: item in s and len(s) == 1)]
        for _, r in hits.iterrows():
            for c in r["consequents"]:
                if c in owned:
                    continue
                if c not in best or r["lift"] > best[c][1]:
                    best[c] = (round(float(r["confidence"]), 2), round(float(r["lift"]), 2))
    ranked = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)[:top_k]
    return [(name, conf, lift) for name, (conf, lift) in ranked]


def best_sellers(top_k=3, exclude=()):
    """Cold-start fallback: when we know NOTHING about the customer (no
    order on file, no product mentioned), recommend the overall most
    frequently purchased items instead of giving up. This has no
    confidence/lift -- it isn't personalized, which the caller should make
    clear to the user rather than presenting it as a targeted match."""
    tx = pd.read_csv(BASE / "data" / "transactions.csv")
    counts = tx["item"].value_counts()
    names = [n for n in counts.index if n not in set(exclude)][:top_k]
    return [(name, None, None) for name in names]


if __name__ == "__main__":
    frequent_itemsets, rules = mine_rules()
    print(f"Frequent itemsets found: {len(frequent_itemsets)}")
    print(f"Association rules found: {len(rules)}\n")

    print("Top 10 rules by lift:")
    top = rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10).copy()
    top["antecedents"] = top["antecedents"].apply(lambda s: ", ".join(s))
    top["consequents"] = top["consequents"].apply(lambda s: ", ".join(s))
    print(top.to_string(index=False))

    for product in ["Wireless Mouse", "Yoga Mat", "Coffee Maker"]:
        print(f"\nCustomers who bought '{product}' also bought:")
        for cons, conf, lift in recommend(product, rules):
            print(f"  -> {cons}  (confidence={conf}, lift={lift})")
