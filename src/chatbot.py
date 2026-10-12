"""
IECSAP - Concept 4: Chatbot (Week 14)
A task-oriented support chatbot that integrates the other concepts into one
dialogue pipeline:

    user message
      -> Multilingual   detect language (script / hint / heuristics), translate to English
      -> NER            ORDER_ID (any form: ORD10053, "order 10053", "10053"), PRODUCT, DATE, MONEY
      -> Order lookup   against data/orders.csv: does it exist, what's in it, what's its status
      -> FAQ retrieval  for general questions (no order ID)
         OR Ticket classification + an order-aware reply built from the REAL order
      -> Market basket  cross-sell from the order contents
      -> LLM phrasing   (optional) rewrites the verified reply; a grounding guard rejects any
                        reply that invents a number/order or drops a required fact
      -> Translate the reply back into the customer's language

Slot-filling + intent-routing over ML components: auditable, and it can never claim
to know about an order that doesn't exist.
"""
import os
import time

from text_classification import TicketClassifier
from ner import TicketNER
from qa_system import FAQRetrievalQA
from multilingual import detect_and_translate_ex, translate_ex
from llm import LLMClient, LLMError
from preprocessing import clean_text, extract_regex_entities, find_order_ids
from orders import OrderStore
from market_basket import mine_rules, recommend, recommend_for_items, best_sellers

# Messages that start like this are treated as questions even without a "?".
QUESTION_STARTERS = (
    "what", "how", "when", "where", "which", "who", "why", "can i", "can you", "could you",
    "do you", "does", "is there", "is it", "are there", "are you", "tell me", "i want to know",
    "i need to know", "i would like to know", "explain",
)
# A statement (not a question) is still sent to the FAQ if it matches an entry this well.
FAQ_STATEMENT_MIN = 0.85
# At/above this similarity the FAQ text is already the verified final answer; sending it
# through the LLM only adds latency and risk. Set FAQ_VERBATIM_ABOVE=2 to always use the LLM.
FAQ_VERBATIM_ABOVE = float(os.environ.get("FAQ_VERBATIM_ABOVE", "0.85"))

CATEGORY_TEMPLATES = {
    "Billing": "I've logged this as a Billing issue{order}. Our billing team will review the charge{amount} and respond within 24 hours.",
    "Shipping": "I've logged this as a Shipping issue{order}. I'll check the courier status{date} and update you shortly.",
    "Product": "I've logged this as a Product issue{order}. If the {product} is faulty we can arrange a replacement.",
    "Account": "I've logged this as an Account issue. For security, please verify your registered email so we can proceed.",
    "Returns": "I've logged this as a Return/Cancellation request{order}. You'll receive pickup instructions within 24 hours.",
}

CATEGORY_ACTIONS = {
    "Billing": "I've flagged the billing concern for our team to review within 24 hours.",
    "Product": "If any item is faulty we can arrange a replacement.",
    "Returns": "I've logged a return/cancellation request; you'll receive instructions within 24 hours.",
    "Account": "For any account or security changes, I'll first need to verify your registered email.",
    "Shipping": "",
    "Order status": "",
}

# The classifier is trained on long ticket templates and is unreliable on very short messages
# ("my order arrived damaged" -> Billing). Unambiguous keywords are checked first.
CATEGORY_KEYWORDS = {
    "Product": ("damaged", "broken", "faulty", "defective", "not working", "stopped working", "wrong item", "scratched", "cracked"),
    "Billing": ("charged", "overcharged", "double charge", "billing", "invoice", "payment failed", "deducted"),
    "Shipping": ("delayed", "late", "not arrived", "hasn't arrived", "where is", "tracking", "not delivered", "not shipped"),
    "Account": ("password", "login", "log in", "sign in", "my account", "email address change"),
    "Returns": ("return", "refund", "exchange", "cancel"),
}


def keyword_category(text: str):
    t = text.lower()
    hits = [c for c, kws in CATEGORY_KEYWORDS.items() if any(k in t for k in kws)]
    return hits[0] if len(hits) == 1 else None


CLARIFY_REPLY = ("Hello! I can look up an order (just share the order ID, like ORD10053), answer questions "
                 "about returns, shipping, refunds and payments, or log a problem for our team. "
                 "What do you need help with?")


def _residual_words(english_text: str, order_ids: list) -> int:
    """How many meaningful words are left once order IDs are removed. A message that is
    just '10053' (or 'hello') has nothing for the ticket classifier to work with."""
    t = english_text
    for oid in order_ids:
        t = t.replace(oid, " ").replace(oid[3:], " ")
    return len(clean_text(t).split())


def order_summary(o: dict) -> str:
    """One factual sentence block built only from the order record."""
    items = ", ".join(o["items"]) if o["items"] else "no items listed"
    return (f"Order {o['order_id']} was placed on {o['order_date']} and its status is {o['status']}. "
            f"It contains: {items}. The total is ${o['total_amount']:.2f}.")


class SupportChatbot:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm
        self.classifier = TicketClassifier()
        self.orders = OrderStore()
        self.ner = TicketNER(known_orders=self.orders.known_ids())
        self.qa = FAQRetrievalQA()
        _, self.rules = mine_rules()
        self._known = self.orders.known_ids()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _looks_like_question(text: str) -> bool:
        t = text.lower().strip()
        return t.endswith("?") or t.startswith(QUESTION_STARTERS)

    def _fill_reply(self, category: str, entities: dict, order_info: dict | None) -> str:
        if order_info:
            # Real order details are ALWAYS included, then a category-specific next step.
            action = CATEGORY_ACTIONS.get(category, "")
            return (order_summary(order_info) + (" " + action if action else "")).strip()
        template = CATEGORY_TEMPLATES.get(category, "Thanks, I've logged your request and a team member will follow up.")
        order = f" for order {entities['ORDER_ID'][0]}" if entities.get("ORDER_ID") else ""
        amount = f" of {entities['AMOUNT'][0]}" if entities.get("AMOUNT") else ""
        date = f" (expected {entities['DATE'][0]})" if entities.get("DATE") else ""
        product = entities["PRODUCT"][0] if entities.get("PRODUCT") else "item"
        return template.format(order=order, amount=amount, date=date, product=product)

    @staticmethod
    def _flatten(recs, owned=()):
        """Rules can have multi-item consequents ("Laptop Stand, Mechanical Keyboard"). Split them into
        single products, drop anything the customer already owns, keep the best lift per product."""
        best = {}
        for name, conf, lift in recs:
            for prod in (p.strip() for p in str(name).split(",")):
                if prod and prod not in owned and (prod not in best or (lift or 0) > (best[prod][1] or 0)):
                    best[prod] = (conf, lift)
        return [(n, c, l) for n, (c, l) in sorted(best.items(), key=lambda kv: -(kv[1][1] or 0))][:3]

    def _get_recommendations(self, order_info: dict | None, entities: dict):
        """Three tiers, most to least personalised:
        1. the customer's REAL order contents  2. a product they MENTIONED  3. best-sellers (cold start)."""
        owned = set(order_info["items"]) if order_info else set()
        if order_info and order_info["items"]:
            recs = self._flatten(recommend_for_items(order_info["items"], self.rules, top_k=6), owned)
            if recs:
                return recs, "order_history"
        product = entities.get("PRODUCT", [None])[0]
        if product:
            recs = self._flatten(recommend(product, self.rules, top_k=6), owned | {product})
            if recs:
                return recs, "mentioned_product"
        return best_sellers(exclude=owned), "best_sellers"

    def _collect_entities(self, raw: str, english: str) -> dict:
        """spaCy/ruler entities from the English text, with order IDs also taken from the
        RAW message (an order ID is language-independent; translation can mangle it)."""
        entities = self.ner.extract(english)
        ids = find_order_ids(raw, self._known)
        for oid in entities.get("ORDER_ID", []):
            if oid not in ids:
                ids.append(oid)
        if ids:
            entities["ORDER_ID"] = ids
        return entities

    # ------------------------------------------------------------------- main
    def handle(self, raw_message: str, source_lang: str | None = None, use_llm: bool = True,
               history: list | None = None) -> dict:
        steps = []

        def stage(name, t0, detail=""):
            steps.append({"stage": name, "ms": round((time.perf_counter() - t0) * 1000, 1), "detail": detail})

        # 1) language + translation
        t0 = time.perf_counter()
        ml = detect_and_translate_ex(raw_message, source_lang)
        english_text, detected_lang, translated_ok = ml["english"], ml["language"], ml["ok"]
        stage("Language & translation", t0,
              f"{detected_lang}" + (f" -> en via {ml['engine']}" if detected_lang != "en" and ml["ok"] else ""))

        if not translated_ok:
            # Honest failure: we can't understand the message, but an order ID is language-independent,
            # so the real order can still be shown while a human takes over.
            ids = find_order_ids(raw_message, self._known)
            entities = {k: v for k, v in extract_regex_entities(raw_message).items() if v}
            order_info = self.orders.lookup(ids[0]) if ids else None
            if order_info:
                note = " " + order_summary(order_info)
            elif ids:
                note = f" I couldn't find an order with ID {ids[0]} in our system."
            else:
                note = ""
            return {
                "detected_language": detected_lang, "english_text": english_text,
                "reply": ("I couldn't translate your message yet, so I've passed it to a human "
                          "agent who can help." + note),
                "route": "ESCALATE", "category": None, "entities": entities,
                "confidence": None, "escalate": True, "translation_ok": False,
                "order": order_info, "recommendations": [], "recommendation_source": None,
                "llm_used": False, "base_reply_en": None, "reply_en": None,
                "translation_engine": None, "reply_translation_ok": True, "steps": steps,
            }

        # 2) entities (incl. order ID in any form)
        t0 = time.perf_counter()
        entities = self._collect_entities(raw_message, english_text)
        order_id = (entities.get("ORDER_ID") or [None])[0]
        stage("Entity extraction", t0, ", ".join(entities) or "none")

        # 3) route: order ID present -> order/ticket path; otherwise FAQ if it is a question
        faq_answer, faq_conf = None, None
        qa_result = None
        if not order_id:
            t0 = time.perf_counter()
            qa_result = self.qa.answer(english_text)
            is_question = self._looks_like_question(english_text)
            use_faq = is_question or (not qa_result["escalate"] and qa_result["confidence"] >= FAQ_STATEMENT_MIN)
            stage("FAQ retrieval", t0, f"similarity {qa_result['confidence']:.2f}" + ("" if use_faq else " (not a question; ignored)"))
        else:
            use_faq = False

        residual = _residual_words(english_text, entities.get("ORDER_ID", []))
        if not order_id and not use_faq and residual < 2:
            # Greeting / one-word message: nothing to classify. Ask instead of mis-filing a ticket.
            reply_en = CLARIFY_REPLY
            meta = {"route": "CLARIFY", "category": None, "entities": entities,
                    "confidence": None, "escalate": False}
            order_info, recommendations, rec_source = None, [], None
        elif use_faq:
            reply_en = qa_result["answer"]
            faq_answer = None if qa_result["escalate"] else qa_result["answer"]
            faq_conf = qa_result["confidence"]
            meta = {"route": "QA", "category": None, "entities": entities,
                    "confidence": qa_result["confidence"], "escalate": qa_result["escalate"]}
            order_info = None
            recommendations, rec_source = [], None
        else:
            t0 = time.perf_counter()
            if order_id and residual < 2:
                category = "Order status"      # bare ID / "tell me about order X": show the order, no ticket
                stage("Ticket classification", t0, "order status request (no complaint text)")
            else:
                category = keyword_category(english_text)
                how = "keyword rule"
                if category is None:
                    category, how = self.classifier.predict(english_text), "classifier"
                stage("Ticket classification", t0, f"{category} ({how})")
            t0 = time.perf_counter()
            order_info = self.orders.lookup(order_id) if order_id else None
            stage("Order lookup", t0, ("found" if order_info else "not found") if order_id else "no order id")

            if order_id and order_info is None:
                reply_en = (f"I couldn't find an order with ID {order_id} in our system. "
                            "Could you double-check the number, or share the email address you ordered with?")
                meta = {"route": "ORDER_NOT_FOUND", "category": category, "entities": entities,
                        "confidence": None, "escalate": False}
                recommendations, rec_source = [], None
            else:
                reply_en = self._fill_reply(category, entities, order_info)
                meta = {"route": "TICKET", "category": category, "entities": entities,
                        "confidence": None, "escalate": False}
                t0 = time.perf_counter()
                recommendations, rec_source = self._get_recommendations(order_info, entities)
                stage("Market basket", t0, rec_source or "")

        recs_out = [{"product": n, "confidence": c, "lift": l} for n, c, l in recommendations]

        # 4) LLM phrasing, grounded and guarded
        base_reply_en = reply_en
        llm_used = False
        if use_llm and self.llm is not None and self.llm.enabled:
            t0 = time.perf_counter()
            if meta["route"] == "QA" and faq_conf is not None and faq_conf >= FAQ_VERBATIM_ABOVE:
                stage("LLM response", t0, f"skipped: verified FAQ answer used as written (similarity {faq_conf:.2f})")
            elif meta["route"] == "CLARIFY":
                stage("LLM response", t0, "skipped: fixed greeting")
            elif meta["route"] == "QA" and meta["escalate"]:
                stage("LLM response", t0, "skipped: no FAQ answer, handing off to a human")
            else:
                fact_sheet = LLMClient.build_fact_sheet(meta["route"], meta["category"], meta["entities"],
                                                        order_info, faq_answer, base_reply_en, recs_out)
                required = []
                if order_info:
                    required = [order_info["order_id"], str(order_info["status"])]
                elif meta["route"] == "ORDER_NOT_FOUND":
                    required = [order_id]
                try:
                    candidate = self.llm.compose_reply(english_text, fact_sheet, history, route=meta["route"])
                    problem = LLMClient.check_reply(candidate, fact_sheet + " " + english_text, required)
                    if problem:
                        stage("LLM response", t0, f"failed grounding check ({problem}); verified reply used")
                    else:
                        reply_en = candidate
                        llm_used = reply_en != base_reply_en
                        stage("LLM response", t0, f"{self.llm.provider} / {self.llm.model}")
                except LLMError as e:
                    stage("LLM response", t0, f"failed, used template ({str(e)[:60]})")

        # 5) translate the reply back
        t0 = time.perf_counter()
        back = translate_ex(reply_en, "en", detected_lang)
        reply = back["text"]
        if detected_lang != "en":
            stage("Translate reply", t0, f"en -> {detected_lang} via {back['engine'] or 'unavailable (shown in English)'}")

        return {
            "detected_language": detected_lang, "english_text": english_text, "reply": reply,
            "translation_ok": True, "reply_translation_ok": bool(back["ok"]), "order": order_info,
            "recommendations": recs_out, "recommendation_source": rec_source,
            "llm_used": llm_used, "base_reply_en": base_reply_en, "reply_en": reply_en,
            "translation_engine": ml["engine"] or back["engine"], "steps": steps,
            **meta,
        }


if __name__ == "__main__":
    bot = SupportChatbot()
    real = bot.orders.df.index[0]
    for msg in [f"My order {real} hasn't shipped yet, what's going on?", real[3:], f"tell me about order {real}",
                "What is your return policy?", "My order ORD00000 arrived damaged.",
                "Do you offer cash on delivery?", "I was charged twice for my last purchase"]:
        r = bot.handle(msg, use_llm=False)
        print(f"\nUser: {msg}\n  route={r['route']} cat={r['category']} order={bool(r['order'])}\n  reply={r['reply']}")
