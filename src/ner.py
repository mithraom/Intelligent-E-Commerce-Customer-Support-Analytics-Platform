"""
IECSAP - Concept 2: Named Entity Recognition (Week 13)
Extracts structured slots from a free-text ticket so the chatbot/backend
can act on it directly: ORDER_ID, PRODUCT, DATE, MONEY, plus spaCy's
general-purpose entities (PERSON, GPE, ORG...) for anything outside our domain.

Approach: spaCy's pretrained statistical pipeline (en_core_web_sm) extended
with a rule-based EntityRuler for domain-specific patterns (order IDs, product
names) that a generic model was never trained to recognise. This hybrid
statistical + rule-based design is standard practice for narrow-domain NER
where labeled training data is scarce.
"""
import spacy
from spacy.pipeline import EntityRuler
from pathlib import Path
import pandas as pd

from preprocessing import find_order_ids

BASE = Path(__file__).resolve().parent.parent


def build_ner_pipeline():
    nlp = spacy.load("en_core_web_sm")

    products = pd.read_csv(BASE / "data" / "transactions.csv")["item"].unique().tolist()

    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = [{"label": "ORDER_ID", "pattern": [{"TEXT": {"REGEX": r"^ORD\d{5}$"}}]}]
        patterns += [{"label": "PRODUCT", "pattern": [{"LOWER": w.lower()} for w in p.split()]} for p in products]
        ruler.add_patterns(patterns)
    return nlp


class TicketNER:
    def __init__(self, known_orders=None):
        self.nlp = build_ner_pipeline()
        self.known_orders = set(known_orders or [])

    def extract(self, text: str) -> dict:
        doc = self.nlp(text)
        out = {}
        for ent in doc.ents:
            if ent.label_ in ("CARDINAL", "ORDINAL"):   # noise: "about order", "twice", "first"...
                continue
            out.setdefault(ent.label_, []).append(ent.text)

        # The spaCy ruler only knows the exact form "ORD12345". Customers also
        # write "10053", "order 10053", "#10053"; find_order_ids() handles those
        # and returns canonical IDs, which replace the ruler's (possibly partial) hits.
        ids = find_order_ids(text, self.known_orders)
        if ids:
            out["ORDER_ID"] = ids
            digits = {i[3:] for i in ids}
            # drop the bare number if spaCy also tagged it as CARDINAL/MONEY/etc.
            for label in list(out):
                if label == "ORDER_ID":
                    continue
                out[label] = [v for v in out[label] if v not in digits]
                if not out[label]:
                    del out[label]
        return out


def evaluate_on_sample(n=15):
    """Qualitative evaluation: run on n real tickets and report entity coverage
    (Week 13 evaluation criteria: integration, result accuracy)."""
    df = pd.read_csv(BASE / "data" / "tickets.csv").sample(n, random_state=1)
    ner = TicketNER()
    hits, total = 0, 0
    for _, row in df.iterrows():
        ents = ner.extract(row["text"])
        found_order = "ORD" in row["text"] and "ORDER_ID" in ents
        total += 1 if "ORD" in row["text"] else 0
        hits += 1 if found_order else 0
        print(f"\nText: {row['text']}\nEntities: {ents}")
    if total:
        print(f"\nOrder-ID recall on sample: {hits}/{total} = {hits/total:.2%}")


if __name__ == "__main__":
    evaluate_on_sample()
