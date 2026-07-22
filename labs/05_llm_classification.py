# %% [markdown]
# # Lab 5: An LLM-Assisted Labeling Pipeline
#
# ICPSR 2026 — The Social Science Data Pipeline
# Instructor: Andy Halterman
#
# Yesterday you classified the India Police Events articles two ways: TF-IDF and
# embeddings, both trained on hand labels. Today we do the same job a third way,
# with an LLM and no training data at all. You'll write a codebook as a prompt,
# get structured output back, estimate the cost before running at scale, label a
# sample, and check it against the gold labels.
#
# We stay on the same corpus as Lab 3b on purpose: you already know these
# articles and the label set, so what's new here is the LLM pipeline, not the
# data. Day 4 comes back to these labels and asks how good they actually are.

# %%
# Setup (this same block opens every lab this week)
import os

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IN_COLAB = is_colab()
print(f"Environment detected: {'Colab' if IN_COLAB else 'Local/hosted Jupyter'}")

if IN_COLAB:
    if not os.path.exists("/content/icpsr-pipeline"):
        !git clone -q https://github.com/ahalterman/icpsr-pipeline.git /content/icpsr-pipeline
    COURSE_DIR = "/content/icpsr-pipeline"
else:
    COURSE_DIR = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) in ("labs", "solutions") else os.getcwd()

DATA_DIR = os.path.join(COURSE_DIR, "data", "cached")
OUTPUTS_DIR = os.path.join(COURSE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# %%
# The one new package is the OpenAI client. Despite the name it's a generic
# client for any OpenAI-compatible API, and we point it at OpenRouter.
!pip install -q openai pandas scikit-learn tqdm

# %% [markdown]
# ## Connecting to OpenRouter
#
# You each have an OpenRouter API key (distributed on Day 1, with a spending
# cap). OpenRouter is a broker that sits in front of dozens of model providers
# and speaks the standard OpenAI API format, so the same few lines of code can
# call Qwen, Llama, Claude, or GPT by changing one string.
#
# Never paste an API key into a notebook. We read it from an environment
# variable. If no key is set, the lab runs offline from a cache of saved model
# responses, so a dead connection doesn't stop class. Running the calls live is
# the real exercise; the cache is just a backstop.

# %%
# Set OPENROUTER_API_KEY before starting Jupyter, or paste it in.

HAVE_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))

if not HAVE_KEY:
    import getpass
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Paste your OpenRouter API key: ")

HAVE_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))
print("Live calls enabled." if HAVE_KEY else "No key found -- running offline from the response cache.")

# %%
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
) if HAVE_KEY else None

# Our workhorse model for the week. Qwen3-30B-A3B is a mixture-of-experts model:
# 30B total parameters but only ~3B active per token, so it's fast and cheap
# while still being a capable classifier.
MODEL = "qwen/qwen3-30b-a3b"

# %%
import re
import time
import textwrap

def chat(prompt, model=MODEL, temperature=0.0, max_retries=3):
    """Send one prompt to the model and return the response text.

    temperature=0.0 makes the output (more, but not totally) deterministic, which is what you
    want for classification. We retry on failure, and we strip <think>...</think>
    blocks because Qwen3 sometimes reasons out loud before answering.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            text = resp.choices[0].message.content
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

# %%
if HAVE_KEY:
    print(chat("Reply with exactly one word: ready"))
else:
    print("Offline -- skipping the test.")

# %% [markdown]
# ## The data
#
# Same corpus as Lab 3b: 1,257 *Times of India* articles from around the 2002
# Gujarat violence, each hand-labeled for which kinds of police activity it
# reports (Halterman et al. 2021). The five event types:
#
# - `ARREST` — police arrest or detain people
# - `KILL` — police kill someone
# - `FORCE` — police use force (baton charges, tear gas, firing, beating)
# - `FAIL` — police are present but stand by and don't intervene
# - `ANY_ACTION` — the police do *or say* anything at all
#
# We'll spend most of the lab on `ANY_ACTION`, the broadest and most clearly
# topical label, and come back to the rarer ones at the end. Each article can
# carry several labels or none.

# %%
import pandas as pd

df = pd.read_json(os.path.join(DATA_DIR, "india_police_events.jsonl"), lines=True)

event_types = ["ANY_ACTION", "ARREST", "KILL", "FORCE", "FAIL"]
for event in event_types:
    df[event] = df["doc_labels"].apply(lambda labels: 1 if event in labels else 0)

print("Share of articles with each label:")
for event in event_types:
    print(f"  {event:11s} {df[event].mean():.0%}   ({df[event].sum()} articles)")

# %% [markdown]
# We work on a fixed random sample of 150 articles, not all 1,257. That keeps a
# live sequential run to a few minutes and the cost to pennies, and 150
# gold-labeled documents is a much better evaluation set than most real projects
# start with. The sample is deterministic (`random_state=2026`), so everyone in
# the room and the response cache are all working on the same 150 articles.

# %%
work = df.sample(150, random_state=2026).reset_index(drop=True)
print(f"{len(work)} articles | {work['ANY_ACTION'].mean():.0%} are ANY_ACTION")

# %% [markdown]
# ## Step 1: Write the codebook — yours before ours
#
# The chapter's claim is that your prompt is your codebook: the definitions and
# decision rules in it are where the measurement actually happens. So you write
# yours first, before you see ours.
#
# The target is one binary judgment: **does this article report the police doing
# anything?** Read a few articles (below), then take a few minutes to
# fill in the drafting cell: a one-paragraph definition and whatever decision
# rules it turns out to need. You'll hit the boundary cases as you read: an
# article that only quotes a police spokesman, one where the police are the
# victims, one where they stand by and do nothing. Your rules have to settle
# those.

# %%
# Wrap the full text of a few full articles (without truncation) so you can read them.
for _, row in work.sample(4, random_state=7).iterrows():
    text = row["doc_text"]
    print(textwrap.fill(text, width=80))


# %%
# Your draft codebook. Replace every UNWRITTEN placeholder. Keep the JSON output
# line as is -- it's plumbing, not measurement.
MY_CODEBOOK = """You are labeling news articles from the Times of India (2002) for a research project on policing.

[UNWRITTEN: Write your definitions and task description here.]

Answer "yes" if: UNWRITTEN

Answer "no" if: UNWRITTEN

Respond with ONLY a JSON object, no other text:
{"label": "yes" or "no"}

Article:
"""

DRAFTED = "UNWRITTEN" not in MY_CODEBOOK
if not os.environ.get("ICPSR_AUTOMATED"):
    assert DRAFTED, "Write your own definition and rules before running on."

# %%
# Run YOUR codebook on the four sample articles and read what it does.
peek = work.sample(4, random_state=7).reset_index(drop=True)
if HAVE_KEY and DRAFTED:
    for i, row in peek.iterrows():
        raw = chat(MY_CODEBOOK + row["doc_text"][:2000])
        print(f"[{i}] gold ANY_ACTION={row['ANY_ACTION']}  model={raw}")
else:
    print("No key or codebook not drafted -- the text diff below is the half that needs no API.")

# %% [markdown]
# ## The course codebook
#
# Now ours. Same question, same output line; the difference is all in the
# definition and rules. Notice that a police *statement* counts — "police said
# they are investigating" is a `yes`. That follows the original codebook's fifth
# question ("did police do *or say* something else"), and it's the boundary most
# first drafts miss.

# %%
CODEBOOK_PROMPT = """You are labeling news articles from the Times of India (2002) for a research project on policing during the Gujarat violence.

Question: does this article report the police doing or saying anything?

Answer "yes" if the article reports the police as an active agent, e.g.:
- police arresting, detaining, or jailing people
- police killing someone
- police using force (lathi/baton charges, tear gas, firing, beating)
- police raiding, searching, patrolling, imposing curfew, or registering a case
- police investigating, OR making a statement, report, or announcement of any kind

"police" here refers to any government security force, including the army or paramilitary, if they are acting in a policing role.
Keep in mind that the article may refer to police obliquely, e.g., "the men in khaki" or "the authorities".

Answer "no" if:
- the police are not mentioned as an actor, or appear only as victims or background
- the ONLY thing reported is the police failing to act or standing by without responding
- the article is about something else entirely

Code what the article REPORTS, not what you think happened. If you cannot tell, answer "no".

Respond with ONLY a JSON object, no other text:
{"label": "yes" or "no"}

Article:
"""

# %% [markdown]
# The difference between your codebook and the original is potentially interesting. Compare them and
# write down:
#
# - a rule in the original has that yours lacked (police statements? police as victims?);
# - a rule yours has that ours lacks;
# - one article from the four above where you think the two codebooks would
#   disagree, and which one you think is right.

# %%
# Try the course codebook on one article, end to end.
example = work.iloc[0]["doc_text"]
print(textwrap.fill(example, width=80), "...\n")
if HAVE_KEY:
    raw = chat(CODEBOOK_PROMPT + example)
else:
    raw = '{"label": "yes"}'   # a canned response so later steps won't break
print(raw)

# %% [markdown]
# ## Step 2: Parse and validate
#
# Now we can (a) pull the JSON out of the response, (b) expect
# that to fail sometimes and handle it, and (c) check that the label is actually
# from our label set. Valid JSON can still be bad.

# %%
import json

VALID_LABELS = {"yes", "no"}

def parse_response(raw_response):
    """Return (label, explanation) from a model response, or (None, None) on any
    failure. `explanation` is None unless the model included one."""
    if raw_response is None:
        return None, None
    match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)   # grab the {...} block
    if not match:
        return None, None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None
    label = str(obj.get("label", "")).strip().lower()
    if label not in VALID_LABELS:
        return None, None
    return label, obj.get("explanation")

parse_response(raw)

# %%
# Exercise: feed parse_response some malformed responses and confirm it returns
# (None, None) instead of crashing. Try prose with no JSON, a made-up label, and
# JSON wrapped in ```json fences.

# try it here

# %% [markdown]
# ## Step 3: Estimate the cost before you run it
#
# Before any batch run, do the token math. It's easy to skip and easy to get a
# surprising bill for, so it's worth doing every time:
#
# - tokens ≈ words × 1.3 (for English; worse for many other languages)
# - cost = n_docs × (input_tokens × input_price + output_tokens × output_price)
#
# These articles are long (a few hundred words each), so the input dominates.
# Look up the current per-million-token prices for our model on
# openrouter.ai/models and fill them in.

# %%
INPUT_PRICE_PER_M = 0.10    # USD per million input tokens -- CHECK THIS
OUTPUT_PRICE_PER_M = 0.30   # USD per million output tokens -- CHECK THIS

prompt_tokens = len(CODEBOOK_PROMPT.split()) * 1.3
avg_doc_tokens = work["doc_text"].str.split().str.len().mean() * 1.3
output_tokens = 10   # the JSON response is tiny

cost_per_doc = ((prompt_tokens + avg_doc_tokens) * INPUT_PRICE_PER_M
                + output_tokens * OUTPUT_PRICE_PER_M) / 1_000_000
print(f"Estimated cost per article: ${cost_per_doc:.6f}")
print(f"This sample ({len(work)} docs): ${cost_per_doc * len(work):.4f}")
print(f"The full corpus (1,257 docs):  ${cost_per_doc * 1257:.4f}")
print(f"A 100,000-doc corpus:          ${cost_per_doc * 100_000:.2f}")

# %% [markdown]
# At what corpus size does it become worth fine-tuning a small model you can run
# yourself for free? We'll come back to this.

# %% [markdown]
# ## Step 4: Run the zero-shot classifier over the sample
#
# "Zero-shot" means instructions only, no worked examples. We loop over the 150
# articles one at a time. This is the slow part (a few minutes live), because
# we're doing it the simple way, one article at a time. The concurrency section
# at the end does the same work much faster.

# %%
from tqdm.auto import tqdm

def classify(texts, prompt_template):
    """Label a list of articles. Returns a list of (label, explanation)."""
    out = []
    for text in tqdm(texts):
        try:
            raw = chat(prompt_template + text)
        except Exception as e:
            print(f"Hard failure, recording None: {e}")
            raw = None
        out.append(parse_response(raw))
    return out

# Offline (no key) or in the course's re-render harness (ICPSR_SMOKE_TEST), take
# labels from the cache instead of a full live run. The student default, with a
# key and no flag, runs everything live.
SMOKE_TEST = bool(os.environ.get("ICPSR_SMOKE_TEST"))
USE_CACHE = SMOKE_TEST or not HAVE_KEY
if USE_CACHE:
    cache = pd.read_parquet(os.path.join(DATA_DIR, "lab5_india_labels_cache.parquet")).set_index("doc_id")
    assert set(work["doc_id"]).issubset(cache.index), \
        "cache is stale -- regenerate data/cached/lab5_india_labels_cache.parquet"
    cache = cache.loc[work["doc_id"]]   # align to our sample order

def label_column(colname, prompt_template):
    """Return the yes/no labels for `work`, live or from the cache."""
    if USE_CACHE:
        return cache[colname].tolist()
    return [lab for lab, _ in classify(work["doc_text"].tolist(), prompt_template)]

work["pred_zeroshot"] = label_column("pred_zeroshot", CODEBOOK_PROMPT)
print(f"Parse failures: {work['pred_zeroshot'].isna().sum()} / {len(work)}")
work[["doc_text", "ANY_ACTION", "pred_zeroshot"]].head(10)

# %% [markdown]
# ## Step 5: Evaluate against the gold labels
#
# Accuracy is the headline, but with a 36% base rate it's a weak one: a
# classifier that always says "no" scores 64%. So we also report precision and
# recall for the "yes" class, exactly as in yesterday's embeddings/classification lab.

# %%
from sklearn.metrics import classification_report, f1_score

ev = work[work["pred_zeroshot"].notna()].copy()
ev["pred01"] = (ev["pred_zeroshot"] == "yes").astype(int)

acc = (ev["ANY_ACTION"] == ev["pred01"]).mean()
print(f"Zero-shot accuracy: {acc:.2%}  (n={len(ev)})")
print(f"F1 (yes = police action): {f1_score(ev['ANY_ACTION'], ev['pred01']):.3f}\n")
print(classification_report(ev["ANY_ACTION"], ev["pred01"], target_names=["no action", "action"]))

# %%
# The confusion matrix: rows = gold, columns = model.
pd.crosstab(ev["ANY_ACTION"], ev["pred01"], rownames=["gold"], colnames=["model"])

# %%
# Read the disagreements. This is where you learn whether the model is wrong or
# your CODEBOOK is wrong. Look at both directions separately.
false_neg = ev[(ev["ANY_ACTION"] == 1) & (ev["pred01"] == 0)]   # gold action, model missed it
false_pos = ev[(ev["ANY_ACTION"] == 0) & (ev["pred01"] == 1)]   # model said action, gold says none
print(f"{len(false_neg)} missed (false negatives), {len(false_pos)} false alarms (false positives)\n")

print("=== A few MISSED action articles: ===\n")
for _, row in false_neg.head(3).iterrows():
    print(textwrap.fill(row["doc_text"], width=80), "...\n")

# %% [markdown]
# For each disagreement, ask: did the model miss a police action stated plainly
# (its mistake), or is this a boundary the codebook never settled (yours)? A
# common pattern here is the article whose only police content is a *statement* —
# the codebook says that counts, and a reader might not.

# %% [markdown]
# ## Step 6: Ask for an explanation, then the label
#
# A cheap trick that often helps: instead of asking for the bare label, ask the
# model to write one sentence of reasoning *first* and the label *second*. Two
# things happen. The reasoning conditions the answer, so accuracy usually ticks
# up (this is a tiny version of "chain of thought"). And the explanations are a
# window into your codebook: on the disagreements, the model tells you *why* it
# decided the way it did, which is often where your definition was unclear.
#
# The model writes left to right, so the explanation has to come before the
# label in the JSON for the reasoning to actually inform it. Put the label first
# and you've just asked it to rationalize a snap judgment.

# %%
COT_PROMPT = CODEBOOK_PROMPT.replace(
    '{"label": "yes" or "no"}',
    '{"explanation": "<one sentence: what police action, if any, the article reports>", "label": "yes" or "no"}',
)

if USE_CACHE:
    work["pred_cot"] = cache["pred_cot"].tolist()
    cot_explanations = cache["expl_cot"].tolist()
else:
    cot = classify(work["doc_text"].tolist(), COT_PROMPT)
    work["pred_cot"] = [lab for lab, _ in cot]
    cot_explanations = [expl for _, expl in cot]

ev = work[work["pred_zeroshot"].notna() & work["pred_cot"].notna()].copy()
for col in ["pred_zeroshot", "pred_cot"]:
    p = (ev[col] == "yes").astype(int)
    print(f"{col:14s} accuracy {(ev['ANY_ACTION'] == p).mean():.2%}  F1 {f1_score(ev['ANY_ACTION'], p):.3f}")

# %%
# Read the model's reasoning on a few articles it got wrong. This is the payoff:
# it usually names the exact codebook ambiguity.
work["expl_cot"] = cot_explanations
wrong = work[(work["pred_cot"].notna()) &
             ((work["pred_cot"] == "yes") != (work["ANY_ACTION"] == 1))]
for _, row in wrong.head(4).iterrows():
    print(f"gold ANY_ACTION={row['ANY_ACTION']}  model={row['pred_cot']}")
    print(f"  reason: {row['expl_cot']}")
    print(f"  {textwrap.fill(row['doc_text'], width=80)} ...\n")

# %% [markdown]
# ## Step 7: Few-shot
#
# Now we add a few worked examples to the prompt. Pick examples near the
# boundary — the hard calls — because easy examples teach the model nothing. The
# examples must NOT come from articles we evaluate on (why not?). These four are
# written by hand in the spirit of the corpus.

# %%
FEW_SHOT_BLOCK = """Here are some labeled examples:

Article: "Police arrested twelve people in connection with the riot and registered cases against them."
{"label": "yes"}

Article: "A police spokesman said the situation was under control and that patrols had been stepped up."
{"label": "yes"}

Article: "Residents alleged that police stood by as the mob attacked shops and did nothing to stop it."
{"label": "no"}

Article: "The chief minister addressed the assembly on the law-and-order situation in the state."
{"label": "no"}

"""

FEWSHOT_PROMPT = CODEBOOK_PROMPT.replace("Article:\n", FEW_SHOT_BLOCK + "Article:\n")

work["pred_fewshot"] = label_column("pred_fewshot", FEWSHOT_PROMPT)

ev = work[work["pred_zeroshot"].notna() & work["pred_fewshot"].notna()].copy()
for col in ["pred_zeroshot", "pred_fewshot"]:
    p = (ev[col] == "yes").astype(int)
    print(f"{col:14s} accuracy {(ev['ANY_ACTION'] == p).mean():.2%}  F1 {f1_score(ev['ANY_ACTION'], p):.3f}")

# %% [markdown]
# With a sample this size the difference may be noise, which is itself a lesson
# for Day 4: how big does an evaluation set have to be before you can tell two
# classifiers apart?

# %% [markdown]
# ## Step 8: The rarer labels
#
# `ANY_ACTION` is the easy case: broad and topical. The specific event types are
# rarer and subtler, and that's where a classifier usually struggles. We reuse
# the same machinery with a different question for each, using the definitions
# and examples from the original codebook. `FAIL` in particular — the police
# present but *not* acting — is a hard call even for people.

# %%
CONSTRUCTS = {
    "ARREST": ("Did the police arrest, detain, or jail anyone?",
               "Police arrested ten people yesterday."),
    "KILL":   ("Did the police kill anyone?",
               "Two people died due to police firing."),
    "FORCE":  ("Did the police use force or violence, such as beating, shooting, tear gas, or a lathi charge?",
               "Police beat innocent bystanders."),
    "FAIL":   ("Were the police present but failed to act or intervene?",
               "The police observed the conflict but did not intervene."),
}

def build_prompt(question, example):
    return f"""You are labeling news articles from the Times of India (2002) for a research project on policing.

Question: {question}

Answer "yes" if the article reports this, e.g.: "{example}"
Answer "no" otherwise. Code what the article REPORTS; if you cannot tell, answer "no".

Respond with ONLY a JSON object, no other text:
{{"label": "yes" or "no"}}

Article:
"""

for event, (question, example) in CONSTRUCTS.items():
    col = f"pred_{event}"
    work[col] = label_column(col, build_prompt(question, example))

# %%
# Precision/recall/F1 for each event type, on the articles that parsed.
print(f"{'event':11s} {'n_gold':>6s} {'precision':>10s} {'recall':>8s} {'F1':>6s}")
from sklearn.metrics import precision_recall_fscore_support
for event in event_types:
    col = f"pred_{event}" if event != "ANY_ACTION" else "pred_zeroshot"
    sub = work[work[col].notna()]
    p = (sub[col] == "yes").astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        sub[event], p, average="binary", zero_division=0)
    print(f"{event:11s} {sub[event].sum():6d} {prec:10.2f} {rec:8.2f} {f1:6.2f}")

# %% [markdown]
# Which labels does the model handle well, and which does it fall down on? Tie
# this back to yesterday: TF-IDF and embeddings also struggled most on the rare
# events, for a related reason (the signal is one sentence in a long article).

# %% [markdown]
# ## Step 9: A taste of prompt sensitivity
#
# A real measurement instrument shouldn't change its answer when you change
# something that doesn't affect the meaning. Let's check. Here is the same
# codebook with the "yes" and "no" blocks swapped in order and lightly reworded —
# same rules, different presentation.

# %%
REWORDED_PROMPT = """You are labeling news articles from the Times of India (2002) for a research project on policing during the Gujarat violence.

Question: does this article report the police doing anything?

Answer "no" if the police are absent, appear only as victims or background, or if the only thing reported is the police standing by without acting. Answer "no" if the article is about something else.

Otherwise answer "yes": the police are an active agent -- arresting or detaining people, killing or using force (lathi charges, tear gas, firing, beating), raiding, searching, patrolling, imposing curfew, registering a case, investigating, or making any statement, report, or announcement.

Code what the article REPORTS, not what you think happened. If you cannot tell, answer "no".

Respond with ONLY a JSON object, no other text:
{"label": "yes" or "no"}

Article:
"""

work["pred_reworded"] = label_column("pred_reworded", REWORDED_PROMPT)

both = work[work["pred_zeroshot"].notna() & work["pred_reworded"].notna()]
flipped = (both["pred_zeroshot"] != both["pred_reworded"])
print(f"Labels changed by rewording: {flipped.sum()} / {len(both)} ({flipped.mean():.1%})")
# A perfect instrument flips 0%. Hold this number for Day 4, where we measure
# prompt sensitivity systematically.

# %% [markdown]
# ## Step 10: Save everything, including the prompt
#
# The labels are only half the output. For reproducibility the prompt, the model
# string, and the date are *part of your data*, so we save them together.

# %%
import datetime

run_metadata = {
    "model": MODEL,
    "temperature": 0.0,
    "codebook_prompt": CODEBOOK_PROMPT,
    "fewshot_prompt": FEWSHOT_PROMPT,
    "run_date": datetime.datetime.now().isoformat(),
}

work.to_parquet(os.path.join(OUTPUTS_DIR, "lab5_labels.parquet"))
with open(os.path.join(OUTPUTS_DIR, "lab5_run_metadata.json"), "w") as f:
    json.dump(run_metadata, f, indent=2)

print("Saved labels and run metadata.")

# %% [markdown]
# ## Capstone variant
#
# Apply this pipeline to your own measurement target:
#
# 1. Write a codebook-prompt for your construct (definition, decision rules, a
#    "cannot determine" path, JSON output line).
# 2. Hand-label 20 documents from your data yourself, *before* running the model.
#    (Yes, before. Why?)
# 3. Run zero-shot, evaluate against your 20, and read every disagreement.
# 4. Bring to stand-up: one disagreement where the model was wrong, and one where
#    your codebook was wrong.

# %% [markdown]
# ## If you finish early
#
# - Swap `MODEL` for a smaller model (`qwen/qwen3-8b`) and a larger one, same
#   prompt. How do accuracy and cost move? Is the expensive model worth it?
# - Add a `"quote"` field to the JSON asking the model to return the phrase that
#   justified its label, then spot-check: do the quotes actually support the
#   labels? A cheap and surprisingly effective audit.

# %% [markdown]
# ## Advanced: run it concurrently
#
# (Optional, and better after the scaling session.) Step 4 labeled the articles
# one at a time. Almost all of that time was spent waiting on the network, not
# computing, so we can send many requests at once and go much faster. Threads are
# the simple version and are fine at this scale; `asyncio` (the scaling chapter's
# pattern) is what scales to hundreds of thousands of documents.

# %%
if HAVE_KEY:
    from concurrent.futures import ThreadPoolExecutor

    def classify_one(text):
        return parse_response(chat(CODEBOOK_PROMPT + text[:2000]))

    t0 = time.perf_counter()
    seq = [classify_one(t) for t in work["doc_text"].tolist()[:20]]
    seq_secs = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:   # 8 at a time is polite
        par = list(pool.map(classify_one, work["doc_text"].tolist()[:20]))
    par_secs = time.perf_counter() - t0

    agree = sum(a == b for a, b in zip(seq, par))
    print(f"sequential: {seq_secs:5.1f}s   threads(8): {par_secs:5.1f}s   "
          f"speedup: {seq_secs / par_secs:.1f}x   labels agree: {agree}/20")
else:
    print("No key -- skipping the concurrency demo (it needs live calls).")

# %% [markdown]
# Same labels, same token cost; only the wall-clock changed. The speedup is real
# but usually short of the arithmetic ideal, because a provider caps how many
# requests it will actually run at once. That's why it's worth timing the run
# instead of trusting the arithmetic.
