# IECSAP — Intelligent E-Commerce Customer Support & Analytics Platform

NLP Laboratory Project (22MDCEL2). Six classical NLP concepts working as one system, now extended with
**speech-to-text, translation, a customer feedback loop, and LLM integration**, behind a redesigned web console.

## What it does

A customer writes **or speaks** in their own language. The system detects the language, translates it, works out
what the problem is, pulls the real order from the order database, finds the right answer, optionally has an LLM
phrase it naturally from verified facts, translates the reply back, and lets the customer **rate it**. Every step is
visible in the console's pipeline inspector.

```
 voice ──► Speech-to-Text ─┐
 text  ────────────────────┴─► Language ID ─► Translate→EN ─► NER ─► Classifier / FAQ-QA
                                                                          │
              Feedback ◄─ reply ◄─ Translate→user lang ◄─ LLM phrasing ◄─ Order lookup + Market basket
```

## Concepts implemented

| # | Concept | Module | Status |
|---|---|---|---|
| 1 | Text classification | `src/text_classification.py` | original |
| 2 | Named entity recognition | `src/ner.py` | original |
| 3 | Question answering (FAQ retrieval) | `src/qa_system.py` | original |
| 4 | Chatbot (orchestrates everything) | `src/chatbot.py` | extended: timings, LLM, language hints |
| 5 | Multilingual NLP / translation | `src/multilingual.py` | extended: any-to-any translation, LLM fallback |
| 6 | Market basket analysis | `src/market_basket.py` | original |
| 7 | **Speech-to-text** | `src/speech.py` + `web/app.js` | **new** |
| 8 | **Feedback collection & analysis** | `src/feedback.py` | **new** |
| 9 | **LLM integration (grounded generation)** | `src/llm.py` | **new** |

## Quick start

```bash
cd iecsap
py -m pip install -r requirements.txt
py -m spacy download en_core_web_sm
py src/make_data.py              # first time only: generate datasets
py src/text_classification.py    # first time only: train the classifier
cd server
py app.py
```
Open **http://127.0.0.1:8000**. One command runs everything: the API and the web console. There is no Node.js, no build
step, and the fonts and Chart.js are bundled, so the interface works offline.

The app works with **no API key at all** (replies come from verified templates). Each extra capability is optional:

| Want | Do this |
|---|---|
| AI-phrased replies, LLM translation, hand-off summaries | `copy .env.example .env`, then set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`, or use Ollama) and restart |
| Voice input in Chrome / Edge / Safari | nothing; it uses the browser's built-in recogniser |
| Voice input offline, or in Firefox | `py -m pip install faster-whisper` |
| Offline translation (no LLM) | `py -m pip install transformers sentencepiece torch`, then `cd src && py warm_translation_models.py nllb` and wait for the line `READY` |
| Feedback page | nothing to run: it fills only from ratings real users give (👍/👎 or stars under each reply) |

## The console

- **Support console** — chat with the bot by typing or by pressing the microphone. Pick the language you will speak;
  the transcript is sent with that language as a hint (more reliable than guessing from a short sentence). Beside it, the
  **pipeline inspector** shows each stage with timings, the extracted entities, the order on file, and, when the LLM wrote
  the reply, the verified template next to the final text so you can see what the model changed.
- **Voice & translate** — speak or type in one language, get the translation in another, and hear it read aloud.
  Shows which engine produced it (MarianMT, NLLB-200, the LLM, or the offline phrasebook).
- **Analytics** — ticket volume, sentiment, and the market-basket cross-sell rules.
- **Feedback** — average rating, % satisfied, rating distribution, ratings split by route / LLM vs template /
  typed vs spoken / language, a **review queue** of 1-2 star replies with full context, and CSV export.
- **System** — what is configured and working, with a one-click LLM connection test and an order lookup probe.

## How speech-to-text works

Two engines, so it works on any machine:

1. **Browser engine (default).** The Web Speech API in Chrome / Edge / Safari. The microphone audio is recognised by
   the browser's speech service and only the transcript reaches this server. This engine needs an internet connection
   in Chrome and Edge, because the browser sends the audio to its vendor's recogniser.
2. **Server engine (optional, offline).** The browser records with `MediaRecorder`, uploads to `POST /api/stt`, and
   Whisper (via `faster-whisper`) transcribes it locally. Whisper identifies the language by itself. Set
   `WHISPER_MODEL=small` in `.env` for better accuracy on Tamil and Hindi.

Spoken replies use the browser's text-to-speech. Whether a given language sounds right depends on the voices
installed on the device.

## How the LLM is used, and why it cannot make things up

The classical pipeline decides **what is true** (does the order exist, what is its status, which FAQ answer matches).
The LLM only decides **how to say it**. Each call receives a fact sheet built from the pipeline's output and is told to
use nothing else. If the order was not found, the fact sheet says so, and the model must say so too. If the LLM call
fails or times out, the system silently falls back to the verified template and records that in the pipeline trace.

The LLM does three jobs: phrase the reply, translate when no local translation model is cached, and write a
hand-off summary for a human agent. It also classifies the sentiment of feedback comments (with an offline lexicon
fallback). Providers: Anthropic (default model `claude-haiku-5-5`, change with `LLM_MODEL`), any OpenAI-compatible
endpoint, or Ollama for a fully local model.

## How feedback works

Under each bot reply there are five stars. Clicking one opens a short form with tags (which depend on whether the
rating was high or low) and an optional comment. The rating is stored in SQLite (`data/feedback.db`) together with the
context that produced the reply: route, category, language, typed or spoken, and whether the LLM wrote it. That is
what lets the dashboard answer questions such as "do Tamil voice messages rate lower than English text?" and
"does LLM phrasing score better than templates?". Comments are labelled positive / neutral / negative.

## Run order (command line, without the web app)
```bash
cd src
python make_data.py             # generate orders + tickets + FAQ + transactions
python text_classification.py   # train + evaluate classifier
python ner.py                   # NER evaluation
python qa_system.py             # QA evaluation
python orders.py                # order lookup
python chatbot.py               # chatbot + order lookup + multilingual
python market_basket.py         # association rules
python feedback.py              # feedback store self-test
python demo.py                  # end-to-end demo of the core concepts
```

## Project structure
```
iecsap/
├── data/                  generated datasets (+ feedback.db, created at runtime)
├── src/
│   ├── make_data.py  preprocessing.py  orders.py
│   ├── text_classification.py  ner.py  qa_system.py  market_basket.py
│   ├── chatbot.py          orchestrates the whole pipeline
│   ├── multilingual.py     language ID, translate_ex(), engine fallback chain
│   ├── speech.py           optional server-side Whisper STT
│   ├── llm.py              provider-agnostic LLM client, grounded prompts
│   ├── feedback.py         SQLite feedback store + sentiment + aggregates
│   └── warm_translation_models.py
├── server/app.py          FastAPI backend
├── web/                   index.html, styles.css, app.js, vendor/ (Chart.js + fonts)
├── .env.example
└── requirements.txt
```

## API reference (server/app.py)
| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | `{message, source_lang?, input_mode?, use_llm?, history?}` → reply, route, entities, order, recommendations, per-stage `steps`, `message_id` |
| `/api/stt` | POST | multipart `audio` (+ `language`) → transcript via Whisper; 501 if not installed |
| `/api/translate` | POST | `{text, source ("auto" ok), target}` → translation + engine used |
| `/api/languages` | GET | supported languages |
| `/api/summarize` | POST | `{transcript}` → LLM hand-off summary (needs an LLM) |
| `/api/feedback` | POST | `{message_id, rating 1-5, comment?, tags?}` |
| `/api/feedback/summary` | GET | aggregates for the dashboard |
| `/api/feedback/export` | GET | all feedback as CSV |
| `/api/system` | GET | LLM, STT, translation-model and data status |
| `/api/system/llm-test` | POST | one live round-trip to the LLM |
| `/api/orders/{id}` | GET | direct order lookup |
| `/api/analytics/tickets`, `/api/analytics/rules`, `/api/faq` | GET | dashboard data |

Interactive API docs are served at `/docs`.

## Order lookup, not a placeholder

Earlier versions of this project treated an order ID purely as text: the
chatbot extracted it with NER and just echoed it back, with no way to tell
a real order from a typo or a made-up number. That's fixed. `orders.csv`
(1,200 rows, generated by `make_data.py`) is a real "order-management
system" table: `order_id, customer_id, items, status, order_date,
total_amount`. `orders.py`'s `OrderStore` looks an ID up against it, and the
chatbot branches on what it finds:

- **Order exists** → the reply uses the real status, date and items, and the
  product recommendation is built from everything actually in that order
  (`recommendation_source: "order_history"`).
- **Order ID given but not found** → route `ORDER_NOT_FOUND`. The bot says so
  honestly and asks the customer to double-check the number, rather than
  filing a ticket against — or recommending products based on — an order
  that was never placed. `make_data.py` deliberately makes ~12% of ticket
  order references point at IDs that don't exist, so this path is tested
  against real data, not just handled in theory.
- **No order ID, but a product is mentioned** → recommendations come from
  that product instead (`mentioned_product`).
- **Neither an order nor a product is known (cold start)** → falls back to
  overall best-sellers (`best_sellers`), clearly labeled as not personalized
  rather than presented as a targeted match.

Test it yourself: `GET /api/orders/{order_id}` looks up any ID directly, the
same way the chatbot does internally — useful for confirming a specific ID
is or isn't on file before testing it in the chat.

## Notes on data
Datasets are synthetically generated (`make_data.py`) with realistic
structure: tickets are generated *against* the orders table rather than
independently of it, so a ticket's product mention actually matches
something in that order, and the FAQ and transaction data share the same
co-purchase patterns the Market Basket Analysis is meant to discover. This
keeps the whole pipeline reproducible without external data access.
Swapping in a real helpdesk export / order-management system / FAQ KB
requires no code changes beyond pointing `make_data.py`'s output paths (or
`OrderStore`'s `path` argument) at the real files.

## Notes on Multilingual NLP
`multilingual.py` is written against a `Translator` interface with two
backends:
- `MachineTranslator` — real translation via Hugging Face. It uses small
  MarianMT pairs (`Helsinki-NLP/opus-mt-*`) for common languages, and falls
  back to NLLB-200 (`facebook/nllb-200-distilled-600M`) for languages that
  don't have a small pair, including Tamil and other Indian languages.
- `DemoPhrasebookTranslator` — a tiny offline fallback used only when
  nothing is cached, so the pipeline still responds rather than hanging.

**Translation never happens inside a live chat request.** Both backends run
with `HF_HUB_OFFLINE=1`, so a model that isn't already downloaded fails the
lookup in milliseconds instead of fetching it mid-conversation. If a
non-English message can't be translated, the chatbot says so honestly
(route `ESCALATE`) instead of guessing at its meaning.

To enable real translation for a language, download its model once,
deliberately, with a working internet connection:
```powershell
py -m pip install transformers sentencepiece torch
cd src
py warm_translation_models.py es fr    # Spanish + French (~300MB per direction)
py warm_translation_models.py nllb     # Tamil, Hindi, Telugu, etc. (~2.4GB, one model)
```
After that, `multilingual.py` finds the model in the local cache
automatically — no code changes, and the chat still never blocks on network
mid-request.

## Troubleshooting

**`LookupError: Resource 'wordnet' not found` / the script hangs and you
have to Ctrl-C it, showing `KeyboardInterrupt`.**
This happens on machines where `nltk.download()` tries to reach `nltk.org`
and the network (firewall, proxy, restricted lab network) blocks or
silently drops the connection instead of refusing it quickly, so the
download hangs indefinitely. `preprocessing.py` never calls
`nltk.download()`. It uses NLTK's stopwords/WordNet corpora only if
they're already installed locally, and otherwise transparently falls back
to a bundled stopword list and a lightweight rule-based lemmatizer — no
network access required, ever. Confirm which path is active with:
```bash
cd src
python preprocessing.py   # prints "NLTK WordNet" or "built-in fallback"
```
Both paths give equivalent classifier accuracy on this project's data.

**`'pip' is not recognized` / `'python' is not recognized`.**
On some Windows installs, only the Python *launcher* (`py`) is registered on
PATH. Substitute `py -m pip install ...` for `pip install ...`, and
`py script.py` for `python script.py`. This project's commands above
already use `py` for that reason.

**Chat gets stuck on "…" forever, no reply, no error in the browser console.**
This happened because `langdetect` is unreliable on short text — it once
classified the *English* sentence "My order ORD12345 arrived damaged." as
Danish with 99.99% confidence, which triggered an attempt to download a
translation model mid-request. Fixed two ways in `multilingual.py`: a local
word-overlap check overrides langdetect when enough recognisable English
words are present, and translation runs fully offline (see above) so a live
request can never trigger a download regardless of what language is
detected.

**A customer's message is in a language with no cached model (e.g. Tamil
before running `warm_translation_models.py nllb`).**
The chatbot replies "I couldn't translate your message yet, so I've passed
it to a human agent who can help," keeps any order ID it could still find
with the language-independent regex extractor, and routes to `ESCALATE`.
It does not guess at the message's category or pretend to understand it.

**An order ID in a message doesn't match anything in `orders.csv`.**
This is by design, not a bug — route `ORDER_NOT_FOUND`. The bot asks the
customer to double-check the number rather than filing a ticket or
recommending products against an order that doesn't exist. Check
`GET /api/orders/{id}` to see exactly what the system knows about any ID.
