"""
IECSAP - End-to-end integration demo (Week 15).
Runs a support conversation through the chatbot, which internally covers
all six concepts: Multi-lingual NLP, Text Classification, NER, order lookup
feeding Question Answering / templated replies, and Market Basket Analysis
for the recommendation shown after each ticket. One realistic user journey,
including the two honest-failure paths (order not found, translation not
available) rather than only the happy path.
"""
from chatbot import SupportChatbot

if __name__ == "__main__":
    print("=" * 70)
    print("IECSAP DEMO: Intelligent E-Commerce Customer Support & Analytics")
    print("=" * 70)

    bot = SupportChatbot()
    # Pull one real order ID from the data so this demo is guaranteed to hit
    # the "order found" path at least once, alongside a deliberately fake one.
    real_order_id = bot.orders.df.index[0]

    conversation = [
        f"My order {real_order_id} hasn't shipped yet, what's going on?",
        "What is your return policy?",
        "My order ORD00000 arrived damaged.",          # guaranteed not to exist
        "My Wireless Mouse stopped working, no order number handy.",
        "Hola, mi pedido ORD30012 llego danado.",       # Spanish: translated if a model is cached
    ]

    for msg in conversation:
        result = bot.handle(msg)
        print(f"\nUSER ({result['detected_language']}): {msg}")
        print(f"BOT: {result['reply']}")
        print(f"  [route: {result['route']}"
              + (f" | category: {result['category']}" if result.get("category") else "")
              + f" | entities: {result['entities']}]")
        if result.get("order"):
            o = result["order"]
            print(f"  [order on file: {o['status']}, placed {o['order_date']}, items: {', '.join(o['items'])}]")
        if result.get("recommendations"):
            print(f"  [cross-sell, source={result['recommendation_source']}]")
            for r in result["recommendations"]:
                tag = f"{r['confidence']*100:.0f}% confidence, {r['lift']:.2f}x lift" if r["confidence"] is not None else "popular pick"
                print(f"      -> {r['product']} ({tag})")

    print("\n" + "=" * 70)
    print("All 6 NLP concepts demonstrated: Text Classification, NER, QA,")
    print("Chatbot, Multilingual NLP, Market Basket Analysis.")
    print("=" * 70)
