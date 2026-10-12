"""
IECSAP - Order lookup store.

Backs the chatbot's order-aware replies and recommendations: given an order
ID that NER extracted from a message, this looks up whether it actually
exists, what's in it, and its status -- rather than the chatbot just
echoing the ID back without knowing anything about it (the earlier,
placeholder behaviour).

In production this class would query the real order-management system's
database/API instead of a CSV; nothing else in the project needs to change
to support that, since the chatbot only ever talks to OrderStore.lookup().
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent


class OrderStore:
    def __init__(self, path=None):
        path = path or (BASE / "data" / "orders.csv")
        df = pd.read_csv(path, dtype={"order_id": str, "customer_id": str})
        self.df = df.set_index("order_id")

    def exists(self, order_id: str) -> bool:
        return (order_id or "").strip().upper() in self.df.index

    def known_ids(self) -> set:
        return set(self.df.index)

    def lookup(self, order_id: str):
        """Returns a dict describing the order, or None if it doesn't exist.
        Callers MUST handle the None case explicitly -- that's the whole
        point of this class existing instead of just trusting any ID a
        customer types."""
        order_id = (order_id or "").strip().upper()
        if order_id not in self.df.index:
            return None
        row = self.df.loc[order_id]
        return {
            "order_id": order_id,
            "customer_id": row["customer_id"],
            "items": [i for i in str(row["items"]).split("|") if i],
            "status": row["status"],
            "order_date": row["order_date"],
            "total_amount": float(row["total_amount"]),
        }


if __name__ == "__main__":
    store = OrderStore()
    print(f"Loaded {len(store.df)} orders.")
    real_id = store.df.index[0]
    print("Lookup of a real order     :", store.lookup(real_id))
    print("Lookup of a made-up order  :", store.lookup("ORD00000"))
