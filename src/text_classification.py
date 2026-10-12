"""
IECSAP - Concept 1: Text Classification (Week 12)
Classifies incoming support tickets into a department category
(Billing / Shipping / Product / Account / Returns) and flags sentiment,
so the chatbot/router can send the ticket to the right queue.

Model: TF-IDF (uni+bi-grams) + Linear SVM, compared against Logistic Regression
and Multinomial Naive Bayes as baselines. Evaluated with accuracy, macro-F1,
and a confusion matrix (Week 12/13 evaluation criteria: reliability, accuracy).
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from preprocessing import clean_text

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "outputs"
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(BASE / "data" / "tickets.csv")
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def train_and_evaluate():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }

    results = {}
    best_name, best_model, best_f1 = None, None, -1
    for name, model in models.items():
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        results[name] = {"accuracy": acc, "macro_f1": f1}
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}  Macro-F1: {f1:.3f}")
        print(classification_report(y_test, preds))
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    print(f"\nBest model: {best_name} (macro-F1={best_f1:.3f})")
    cm = confusion_matrix(y_test, best_model.predict(Xte), labels=sorted(df["category"].unique()))
    print("Confusion matrix (rows=true, cols=pred), labels:", sorted(df["category"].unique()))
    print(cm)

    joblib.dump({"vectorizer": vectorizer, "model": best_model, "model_name": best_name},
                MODEL_DIR / "ticket_classifier.joblib")
    return results, best_name


class TicketClassifier:
    """Wrapper used by chatbot.py at inference time."""
    def __init__(self, path=MODEL_DIR / "ticket_classifier.joblib"):
        bundle = joblib.load(path)
        self.vectorizer = bundle["vectorizer"]
        self.model = bundle["model"]

    def predict(self, text: str) -> str:
        vec = self.vectorizer.transform([clean_text(text)])
        return self.model.predict(vec)[0]


if __name__ == "__main__":
    train_and_evaluate()
    clf = TicketClassifier()
    for sample in [
        "My order ORD55123 was never delivered and tracking hasn't moved in a week.",
        "I was double charged $89.99 for order ORD10021, please refund.",
        "The bluetooth headphones I got stopped pairing after one day.",
    ]:
        print(f"\n'{sample}' -> {clf.predict(sample)}")
