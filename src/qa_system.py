"""
IECSAP - Concept 3: Question Answering System (Week 13)
Retrieval-based QA over the company FAQ knowledge base: given a free-text
customer question, return the most relevant FAQ answer plus a confidence
score, with a fallback to "escalate to human agent" below a threshold.

Approach: TF-IDF + cosine similarity over FAQ questions (fast, fully offline).
Two views are blended:
  * word 1-2-grams   - precise when the wording matches
  * character 3-5-grams - tolerant of typos and of the small word-choice
    differences machine translation introduces. This matters because
    non-English questions reach this module as *translated* English, which is
    rarely word-for-word what an English speaker would type.
"""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text

BASE = Path(__file__).resolve().parent.parent

WORD_WEIGHT, CHAR_WEIGHT = 0.6, 0.4


class FAQRetrievalQA:
    def __init__(self, threshold: float = 0.35):
        self.df = pd.read_csv(BASE / "data" / "faq.csv")
        self.threshold = threshold
        corpus = self.df["question"].apply(clean_text)
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2))
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.word_matrix = self.word_vec.fit_transform(corpus)
        self.char_matrix = self.char_vec.fit_transform(corpus)

    def _scores(self, query: str):
        q = clean_text(query)
        w = cosine_similarity(self.word_vec.transform([q]), self.word_matrix).flatten()
        c = cosine_similarity(self.char_vec.transform([q]), self.char_matrix).flatten()
        return WORD_WEIGHT * w + CHAR_WEIGHT * c

    def answer(self, query: str) -> dict:
        sims = self._scores(query)
        best_idx = int(sims.argmax())
        score = float(sims[best_idx])
        if score < self.threshold:
            return {"answer": "I'm not fully sure -- let me connect you with a human agent.",
                    "matched_question": None, "confidence": round(score, 3), "escalate": True}
        return {
            "answer": self.df.iloc[best_idx]["answer"],
            "matched_question": self.df.iloc[best_idx]["canonical_question"],
            "confidence": round(score, 3),
            "escalate": False,
        }


def evaluate():
    """Held-out paraphrases the index has never seen: is the correct ANSWER retrieved?"""
    qa = FAQRetrievalQA()
    unseen_queries = [
        ("I want to send this product back, how?", "Items can be returned within 30 days of delivery if unused and in original packaging."),
        ("How many days for the order to reach me?", "Standard shipping takes 5-7 business days; express shipping takes 2-3 business days."),
        ("Do you deliver internationally?", "Yes, we ship to over 40 countries; international delivery takes 10-15 business days."),
        ("Login isn't working, forgot my password", "Click 'Forgot Password' on the login page and follow the link sent to your registered email."),
        ("Is COD available?", "Cash on delivery is available for orders under $200 in eligible pin codes."),
        ("How do I get my refund faster?", "Refunds are processed within 5-7 business days after the returned item passes inspection."),
        ("Tell me about your return policy", "Items can be returned within 30 days of delivery if unused and in original packaging."),
        ("What is the time to get a refund?", "Refunds are processed within 5-7 business days after the returned item passes inspection."),
        ("Which payment options do you accept?", "We accept credit/debit cards, UPI, net banking, and popular digital wallets."),
        ("can i edit my shipping address", "You can change the address before the order ships, from 'My Orders' > 'Edit Address'."),
    ]
    correct = 0
    for query, expected_answer in unseen_queries:
        result = qa.answer(query)
        ok = result["answer"] == expected_answer
        correct += ok
        print(f"Q: {query}\n -> matched: '{result['matched_question']}' (conf={result['confidence']}) {'OK' if ok else 'MISS'}\n")
    print(f"Accuracy on unseen paraphrases: {correct}/{len(unseen_queries)} = {correct/len(unseen_queries):.2%}")


if __name__ == "__main__":
    evaluate()
