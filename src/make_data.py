"""
IECSAP - Data Collection & Preprocessing (Week 11)
Generates four realistic datasets for an e-commerce customer-support platform:
  1. orders.csv          -> the "order-management system": order -> customer, items, status, date, total
  2. tickets.csv          -> raw customer support messages (for Text Classification + NER)
  3. faq.csv              -> knowledge base (for the QA System)
  4. transactions.csv     -> basket-level purchase history (for Market Basket Analysis)

Tickets are generated AGAINST the orders table, not independently of it:
most tickets reference a real order ID (so the chatbot can look it up and
answer with the real status/items), and a deliberate ~12% reference an
order ID that looks valid but does not exist -- so the system is forced to
handle "I can't find that order" honestly instead of just assuming every
ID a customer types is real.

In production these would be pulled from a helpdesk export (Zendesk/Freshdesk CSV),
the order-management system, and the company FAQ/knowledge base. Here we generate
a synthetic-but-structurally-realistic version so the pipeline is fully reproducible
without external data-access dependencies.
"""
import random
import pandas as pd
from pathlib import Path

random.seed(42)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PRODUCTS = ["Wireless Mouse", "Bluetooth Headphones", "Laptop Stand", "USB-C Charger",
            "Mechanical Keyboard", "Smart Watch", "Yoga Mat", "Running Shoes",
            "Coffee Maker", "Blender", "Desk Lamp", "Backpack", "Water Bottle",
            "Phone Case", "Screen Protector", "Power Bank", "Webcam", "Notebook Set"]

# Approximate unit prices, used to generate realistic order totals.
PRICES = {
    "Wireless Mouse": 18, "Bluetooth Headphones": 45, "Laptop Stand": 25,
    "USB-C Charger": 15, "Mechanical Keyboard": 55, "Smart Watch": 90,
    "Yoga Mat": 20, "Running Shoes": 65, "Coffee Maker": 40, "Blender": 35,
    "Desk Lamp": 22, "Backpack": 38, "Water Bottle": 12, "Phone Case": 10,
    "Screen Protector": 8, "Power Bank": 28, "Webcam": 32, "Notebook Set": 9,
}

# Shared co-purchase structure: used by both make_orders() and
# make_transactions() so the order contents and the basket data that
# Market Basket Analysis learns from are drawn from the same real-world
# pattern (mouse+keyboard+stand, yoga mat+bottle+shoes, etc).
AFFINITY_SETS = [
    ["Wireless Mouse", "Mechanical Keyboard", "Laptop Stand"],
    ["Bluetooth Headphones", "Power Bank", "Phone Case"],
    ["Yoga Mat", "Water Bottle", "Running Shoes"],
    ["Coffee Maker", "Blender"],
    ["Webcam", "Desk Lamp", "Notebook Set"],
    ["Screen Protector", "Phone Case"],
    ["Smart Watch", "Power Bank"],
]

ORDER_STATUSES = ["Processing", "Shipped", "Out for Delivery", "Delivered", "Cancelled", "Delayed"]
ORDER_STATUS_WEIGHTS = [0.12, 0.18, 0.10, 0.45, 0.05, 0.10]

# The fixed universe of order IDs that actually exist. Tickets deliberately
# reference IDs outside this set sometimes (see make_tickets), to exercise
# the "order not found" path honestly rather than only ever succeeding.
_ORDER_ID_POOL = random.sample(range(10000, 99999), 1200)
EXISTING_ORDER_IDS = {f"ORD{n}" for n in _ORDER_ID_POOL}

CATEGORIES = {
    "Billing": [
        "I was charged twice for order {oid}, please refund the extra amount.",
        "My invoice for order {oid} shows the wrong tax amount.",
        "Can I get a refund of {amt} for order {oid}, the item never arrived?",
        "Why was my card charged {amt} when the listed price was different?",
        "I need a GST invoice for order {oid} placed on {date}.",
    ],
    "Shipping": [
        "My order {oid} has not been delivered yet, it was expected on {date}.",
        "The tracking for order {oid} hasn't updated in 5 days.",
        "Can you expedite shipping for order {oid}? I need it before {date}.",
        "I received the wrong address confirmation for order {oid}.",
        "Order {oid} shows delivered but I never received the package.",
        "What happened to order {oid}?",
        "What's going on with order {oid}?",
        "Where is my order {oid}?",
        "Any update on order {oid}?",
    ],
    "Product": [
        "The {product} I received in order {oid} is damaged.",
        "Is the {product} compatible with my old charger?",
        "The {product} stopped working after 2 days, order {oid}.",
        "Does the {product} come with a warranty card?",
        "I want to know the battery life of the {product}.",
    ],
    "Account": [
        "I can't log into my account, it says password incorrect.",
        "Please update my email address on file.",
        "I want to delete my account and all associated data.",
        "My account was locked after too many login attempts.",
        "How do I change my registered phone number?",
    ],
    "Returns": [
        "I want to return the {product} from order {oid}, it doesn't fit.",
        "What is the return window for order {oid}?",
        "The {product} I ordered ({oid}) is not what I expected, how do I exchange it?",
        "Can I cancel order {oid}? It hasn't shipped yet.",
        "Return pickup for order {oid} was missed twice.",
    ],
}

SENTIMENT_NEG_WORDS = ["frustrated", "very disappointed", "unacceptable", "angry", "worst experience"]
SENTIMENT_POS_WORDS = ["thanks so much", "great support so far", "really appreciate it"]


def rand_fake_order_id():
    """An order ID that LOOKS valid but is guaranteed not to exist -- used
    to simulate a customer mistyping a number or asking about an order
    that was never placed."""
    while True:
        oid = f"ORD{random.randint(10000, 99999)}"
        if oid not in EXISTING_ORDER_IDS:
            return oid


def make_orders(n_orders=1200):
    """The 'order-management system': every order the store actually has on
    file, with its items, status, date and total. This is what the chatbot
    looks an order ID up against -- an ID that isn't in here is handled as
    a genuine "order not found" case, not a placeholder."""
    rows = []
    order_ids = sorted(EXISTING_ORDER_IDS)[:n_orders]
    base_date = pd.Timestamp("2026-10-01")
    for oid in order_ids:
        customer_id = f"CUST{random.randint(1000, 9999)}"
        if random.random() < 0.75:
            items = list(dict.fromkeys(random.choice(AFFINITY_SETS)))
            if random.random() < 0.4:
                items.append(random.choice(PRODUCTS))
                items = list(dict.fromkeys(items))
        else:
            items = random.sample(PRODUCTS, k=random.randint(1, 3))
        status = random.choices(ORDER_STATUSES, weights=ORDER_STATUS_WEIGHTS)[0]
        order_date = (base_date - pd.Timedelta(days=random.randint(0, 45))).strftime("%d/%m/%Y")
        total = round(sum(PRICES.get(it, 25) for it in items) * random.uniform(0.95, 1.05), 2)
        rows.append({
            "order_id": oid, "customer_id": customer_id, "items": "|".join(items),
            "status": status, "order_date": order_date, "total_amount": total,
        })
    return pd.DataFrame(rows)


def rand_date():
    return f"{random.randint(1,28)}/{random.randint(1,12)}/2026"


def rand_amount():
    return f"${random.randint(10, 500)}.{random.randint(0,99):02d}"


def make_tickets(orders_df, n=600):
    """Generates tickets AGAINST the real orders table: when a template
    needs an order ID, ~88% of the time it's a real order (so the ticket's
    product mention matches something actually in that order), and ~12% of
    the time it's a fake ID -- the dataset deliberately contains tickets
    about orders that don't exist, so "order not found" is a real,
    measurable case rather than something only handled in theory.
    """
    rows = []
    for i in range(n):
        cat = random.choice(list(CATEGORIES.keys()))
        template = random.choice(CATEGORIES[cat])
        needs_oid = "{oid}" in template

        oid = None
        order_exists = False
        product = random.choice(PRODUCTS)

        if needs_oid:
            if random.random() < 0.88:
                order_row = orders_df.sample(1).iloc[0]
                oid = order_row["order_id"]
                order_exists = True
                order_items = [it for it in order_row["items"].split("|") if it]
                if order_items:
                    product = random.choice(order_items)
            else:
                oid = rand_fake_order_id()
                order_exists = False

        text = template.format(
            oid=oid or rand_fake_order_id(), amt=rand_amount(),
            product=product, date=rand_date()
        )
        # inject sentiment flavor for a subset (used to sanity check classifier robustness)
        if random.random() < 0.25:
            text += " " + random.choice(SENTIMENT_NEG_WORDS) + "."
            sentiment = "negative"
        elif random.random() < 0.4:
            text += " " + random.choice(SENTIMENT_POS_WORDS) + "."
            sentiment = "positive"
        else:
            sentiment = "neutral"
        rows.append({
            "ticket_id": f"T{i+1:05d}", "text": text, "category": cat, "sentiment": sentiment,
            "order_id_used": oid, "order_exists": order_exists if needs_oid else None,
        })
    return pd.DataFrame(rows)


# Each FAQ entry lists several real-world phrasings of the same underlying question,
# mapped to one canonical answer. Indexing paraphrase variants (not just one canonical
# question) is standard production practice for FAQ/retrieval bots and is what makes
# lexical retrieval (TF-IDF) robust to the fact that customers rarely type the FAQ's
# exact wording.
FAQ = [
    (["What is your return policy?", "How can I send items back?", "Can I return something I bought?"],
     "Items can be returned within 30 days of delivery if unused and in original packaging."),
    (["How long does shipping take?", "When will my package arrive?", "What is the delivery time?"],
     "Standard shipping takes 5-7 business days; express shipping takes 2-3 business days."),
    (["How do I track my order?", "Where can I see my delivery status?", "How can I check where my package is?"],
     "You can track your order from the 'My Orders' page using the tracking link emailed to you."),
    (["Do you offer international shipping?", "Do you ship outside the US?", "Can you deliver to other countries?"],
     "Yes, we ship to over 40 countries; international delivery takes 10-15 business days."),
    (["How do I cancel my order?", "Can I stop my order before it ships?"],
     "Orders can be cancelled from 'My Orders' within 1 hour of placement, before they are packed."),
    (["What payment methods are accepted?", "How can I pay for my order?"],
     "We accept credit/debit cards, UPI, net banking, and popular digital wallets."),
    (["How do I reset my password?", "I forgot my password, help.", "I can't remember my login password."],
     "Click 'Forgot Password' on the login page and follow the link sent to your registered email."),
    (["Is there a warranty on electronics?", "Do electronic items come with a guarantee?"],
     "Most electronics carry a 1-year manufacturer warranty; check the product page for details."),
    (["How do I apply a discount code?", "Where do I enter my coupon?"],
     "Enter the code in the 'Promo Code' box at checkout before making payment."),
    (["Can I change my delivery address?", "How do I update my shipping address?"],
     "You can change the address before the order ships, from 'My Orders' > 'Edit Address'."),
    (["What is your refund timeline?", "How soon will I get my money back?"],
     "Refunds are processed within 5-7 business days after the returned item passes inspection."),
    (["Do you offer cash on delivery?", "Can I pay with cash when it arrives?"],
     "Cash on delivery is available for orders under $200 in eligible pin codes."),
]


def make_faq():
    rows = [{"question": q, "answer": ans, "canonical_question": variants[0]}
            for variants, ans in FAQ for q in variants]
    return pd.DataFrame(rows)


def make_transactions(n_baskets=800):
    """Simulate basket-level purchases with the same co-purchase structure
    (AFFINITY_SETS) used for orders.csv, so Apriori/FP-Growth finds rules
    that are consistent with what's actually in the orders table."""
    rows = []
    for i in range(n_baskets):
        if random.random() < 0.75:
            base = random.choice(AFFINITY_SETS)
            basket = set(base)
            if random.random() < 0.4:
                basket.add(random.choice(PRODUCTS))
        else:
            basket = set(random.sample(PRODUCTS, k=random.randint(2, 4)))
        for item in basket:
            rows.append({"basket_id": f"B{i+1:05d}", "item": item})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    orders = make_orders()
    tickets = make_tickets(orders)
    faq = make_faq()
    tx = make_transactions()

    orders.to_csv(DATA_DIR / "orders.csv", index=False)
    tickets.to_csv(DATA_DIR / "tickets.csv", index=False)
    faq.to_csv(DATA_DIR / "faq.csv", index=False)
    tx.to_csv(DATA_DIR / "transactions.csv", index=False)

    print("orders.csv:", orders.shape)
    print(orders["status"].value_counts())

    print("\ntickets.csv:", tickets.shape)
    print(tickets["category"].value_counts())
    referenced = tickets["order_id_used"].notna().sum()
    not_found = ((tickets["order_exists"] == False)).sum()
    print(f"tickets referencing an order: {referenced} ({referenced/len(tickets):.0%})")
    print(f"  of which order does NOT exist: {not_found} ({not_found/referenced:.0%} of those)")

    print("\nfaq.csv:", faq.shape)
    print("\ntransactions.csv:", tx.shape, "| unique baskets:", tx.basket_id.nunique())
