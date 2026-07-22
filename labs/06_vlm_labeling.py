# %% [markdown]
# # Lab 6: Labeling Protest Images with a Vision-Language Model
#
# ICPSR 2026 — The Social Science Data Pipeline
# Instructor: Andy Halterman
#
# This morning you used an LLM to turn text into variables: a codebook written
# as a prompt, structured JSON out, validation against gold. This afternoon we
# do the same measurement job on *images*, with a vision-language model (VLM) —
# an LLM with a vision encoder bolted on, so you can put a photograph in the
# conversation and ask questions about it.
#
# The running example is Steinert-Threlkeld and Joo's MMCHIVED project (the
# paper is in `labs/papers/`). They built a protest event dataset for Chile and
# Venezuela out of images shared on social media, and the images let them
# measure things text struggles with: how big a protest was, whether protesters
# were violent, whether the state was violent. They did it the 2022 way, by
# training a separate CNN for each of those variables and hand-annotating
# thousands of images to do it. We're going to measure the same constructs with
# a VLM and a written codebook, no training data, and then spend real time on
# how the VLM lies to us while doing it.
#
# Two things are different from this morning:
#
# 1. **You'll work on your own images.** There's an upload cell below; drag in
#    photos from your own capstone or research and run the whole codebook on
#    them. Most of the "try it here" cells are written for your images, not ours.
# 2. **We mostly don't have gold.** For a protest photo off the internet there's
#    no answer key. So the validation move shifts from "check against the gold
#    column" to "read the model's evidence against your own eyes."

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
!pip install -q openai pandas pillow tqdm

# %% [markdown]
# ## The model, and what it costs
#
# We use `qwen/qwen3-vl-30b-a3b-instruct` through OpenRouter — the vision
# sibling of the text model you used this morning. Same mixture-of-experts
# recipe (30B parameters, ~3B active per token), plus a vision encoder trained
# to share a representation with the language side. It's priced at **$0.13 per
# million input tokens / $0.52 per million output tokens**.
#
# Images are billed as input tokens, roughly in proportion to resolution: the
# 640-pixel photos here cost a few hundred input tokens each. The whole lab runs
# for well under a cent. That cost is a big part of why VLMs are useful for
# triage, and we'll come back to it at the end.
#
# All the image calls go through `course_utils.chat_image()` — the `chat()` you
# used this morning, plus one image. There's no magic in it; we'll look at what
# it actually does in a minute.

# %%
# Import the course helpers. course_utils.py lives at the repo root, and it is
# course material -- read it, there is nothing hidden in it.
import sys
sys.path.insert(0, COURSE_DIR)

import course_utils
from course_utils import chat_image, parse_json, VISION_MODEL

# Point the call log at the shared outputs/ directory regardless of where this
# notebook runs from. (The log is a Day 5 teaching device.)
course_utils.LOG_PATH = os.path.join(OUTPUTS_DIR, "llm_call_log.csv")

print(f"Vision model: {VISION_MODEL}")

# %%
# This lab is written to be readable WITHOUT an API key: every cell that calls
# the model is guarded by HAVE_KEY, and the markdown after each one describes
# the behavior you'd typically see. If you have your key, set OPENROUTER_API_KEY
# before starting Jupyter, or uncomment the getpass lines below.

# import getpass
# os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Paste your OpenRouter API key: ")

HAVE_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))
if HAVE_KEY:
    print("API key found -- the model cells will run live.")
else:
    print("No OPENROUTER_API_KEY found -- model cells will print a skip note.")
    print("Read the 'typical behavior' notes after each one; the lab still teaches.")

# %% [markdown]
# ## The image set
#
# Our demo images come straight from MMCHIVED's Figure 1 — six photos the
# authors used to illustrate their three image classifiers. They're in
# `data/cached/vlm_protest/`, named so the filename tells you which construct
# each one illustrates and roughly where it sits (a `0` = low/absent, a `1` =
# high/present):
#
# - `protest0.png` / `protest1.png`: **is this a protest?** `protest0` is a
#   soccer player celebrating a goal in front of a stadium crowd — *not* a
#   protest, but the kind of dense-crowd scene a naive protest detector loves to
#   flag. `protest1` is a Chilean street march with hand-lettered health-care
#   signs. This is the false-positive pair.
# - `protestor_violence0.png` / `protestor_violence1.png`: **protester
#   violence.** `0` is people walking past a banner; `1` is masked protesters
#   throwing things at an armored water-cannon truck in tear gas.
# - `state_violence0.png` / `state_violence1.png`: **state violence.** `0` is
#   people holding signs in a Venezuelan street; `1` is a riot-police line with
#   shields and helmets pressed into a crowd.
#
# Six images is enough to *demonstrate* the pipeline, not to *validate* it, and
# that's on purpose. MMCHIVED deliberately does not redistribute its protest
# images (protecting the people in them is a stated ethical commitment of the
# paper), so we ship a tiny licensed sample from the paper's own figure and lean
# on your uploads for volume. These six carry a rough "truth" — the construct
# they were chosen to show — which is the most gold we'll have all afternoon.

# %%
import pandas as pd
from PIL import Image

PROTEST_DIR = os.path.join(DATA_DIR, "vlm_protest")
protest_files = sorted(f for f in os.listdir(PROTEST_DIR) if f.endswith(".png"))
print(protest_files)

# The rough truth these six carry (the construct each was chosen to illustrate).
# Not an answer key -- a low/high designation from the paper's figure. We use it
# to check the model's *ordering*, not to score exact labels.
PAPER_TRUTH = {
    "protest0.png":            {"is_protest": "no",  "note": "soccer celebration, dense crowd -- the false-positive trap"},
    "protest1.png":            {"is_protest": "yes", "note": "Chilean street march, peaceful"},
    "protestor_violence0.png": {"is_protest": "yes", "protester_violence": "low",  "note": "banner + pedestrians, no violence"},
    "protestor_violence1.png": {"is_protest": "yes", "protester_violence": "high", "note": "masked protesters vs. water cannon"},
    "state_violence0.png":     {"is_protest": "yes", "state_violence": "low",  "note": "sign-holders, no police force"},
    "state_violence1.png":     {"is_protest": "yes", "state_violence": "high", "note": "riot-police line pressing a crowd"},
}

# %%
# A quick contact sheet. Spend a minute here before any API call -- decide, with
# your own eyes, what the right answer is for each construct. You can't audit a
# model against a judgment you haven't made yet.
cols, tile = 3, 260
rows = (len(protest_files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tile, rows * tile), "white")
for i, f in enumerate(protest_files):
    im = Image.open(os.path.join(PROTEST_DIR, f)).convert("RGB")
    im.thumbnail((tile - 8, tile - 8))
    sheet.paste(im, ((i % cols) * tile + 4, (i // cols) * tile + 4))
sheet

# %% [markdown]
# ## Upload your own images
#
# Now the part that makes this lab yours. Drop in a handful of images you
# actually care about — protest photos, Telegram channel images, scraped news
# photos, satellite crops, screenshots, whatever your capstone points at. The
# rest of the lab has "try it here" cells written to run on this folder.
#
# In Colab the cell below opens a drag-and-drop widget. Running locally, there's
# no widget; just copy files into the `my_images/` folder it prints and re-run
# the listing cell.

# %%
MYIMG_DIR = os.path.join(OUTPUTS_DIR, "my_images")
os.makedirs(MYIMG_DIR, exist_ok=True)

if IN_COLAB:
    from google.colab import files
    print("Pick one or more image files (drag them onto the button):")
    uploaded = files.upload()  # returns {filename: bytes}
    for name, data in uploaded.items():
        with open(os.path.join(MYIMG_DIR, name), "wb") as fh:
            fh.write(data)
    print(f"\nSaved {len(uploaded)} file(s) to {MYIMG_DIR}")
else:
    print(f"Running locally -- no upload widget. Copy image files into:\n  {MYIMG_DIR}")

# %%
# List whatever is in your folder. Re-run this after uploading more.
def my_images():
    exts = (".png", ".jpg", ".jpeg", ".webp")
    return sorted(os.path.join(MYIMG_DIR, f) for f in os.listdir(MYIMG_DIR)
                  if f.lower().endswith(exts))

my_paths = my_images()
print(f"{len(my_paths)} image(s) in my_images/:")
for p in my_paths:
    print("  ", os.path.basename(p))

# %% [markdown]
# A resolution note that's also a cost and a measurement decision. Big images
# cost more input tokens and give the model more to work with; downscaling saves
# money but throws away detail the model might have needed (a face, a word on a
# sign, a weapon). Our demo images are 640 pixels on the long side: legible and
# cheap. If your uploads are huge phone photos, the helper below shrinks a copy
# before sending. Downscaling is the main cost lever for image pipelines, so
# it's a decision to make deliberately rather than let happen by accident.

# %%
def prepare_image(path, max_side=1024):
    """Return a path to a version of `path` no larger than max_side on its long
    edge (converting to RGB). Small images are returned unchanged."""
    im = Image.open(path).convert("RGB")
    if max(im.size) <= max_side:
        return path
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    out = os.path.join(MYIMG_DIR, "_resized_" + os.path.basename(path))
    im.save(out, "JPEG", quality=90)
    return out

# %% [markdown]
# ## Step 1: One image, one call
#
# Start with the smallest possible thing: hand the model one image and a
# one-line question.

# %%
example_path = os.path.join(PROTEST_DIR, "protest1.png")

if HAVE_KEY:
    answer = chat_image("What is happening in this photo? Reply in one sentence.",
                        example_path)
    print(answer)
else:
    print("[skipped -- no API key]")
    print("Typical answer: one fluent sentence -- 'a group of people marching")
    print("through a city street holding handmade signs, apparently a protest")
    print("or demonstration.' Usually right, and worth checking against the")
    print("image below.")

# %%
# The image we just sent, so you can grade the model's sentence against your
# own eyes. (This audit step never goes away.)
Image.open(example_path)

# %% [markdown]
# ### How the image actually gets there
#
# There's no upload step and no URL. `chat_image()` reads the file, encodes the
# bytes as base64 text, and drops them into the message itself:
#
# ```python
# content = [
#     {"type": "text", "text": prompt},
#     {"type": "image_url",
#      "image_url": {"url": f"data:image/png;base64,{b64}"}},
# ]
# ```
#
# That's the whole trick: the "multimodal API" is the same chat endpoint you've
# used all week, with the image included as a very long base64 string. Open
# `course_utils.py` and confirm there's nothing else going on.

# %% [markdown]
# ## Step 2: The codebook prompt, image edition
#
# A one-line question is fine for a demo. For measurement we need this morning's
# discipline: defined categories, decision rules, an explicit "cannot determine"
# path, and a JSON schema. What's new is that we're asking one image for several
# MMCHIVED variables at once — is it a protest, roughly how big, protester
# violence, state violence.
#
# One field is required today: an **`evidence`** field. Make the model cite the
# *visible* features behind each answer. It doesn't make the model see any
# better, but it gives you something checkable: when the evidence describes a
# water cannon that isn't in the frame, you've caught a confabulation that a bare
# label would have hidden.

# %%
PROTEST_CODEBOOK = """You are coding a photograph for a protest event dataset, working only from what is visible in the image.

Return these fields:

- is_protest: "yes" if the image shows a political protest, demonstration, march, or rally (people gathered to express a collective political demand -- signs, banners, chants, marching). "no" otherwise (sports, concerts, ordinary street scenes, celebrations). "unclear" if you genuinely cannot tell.
- crowd_size: your rough estimate of how many people are visible: "none", "handful" (1-10), "dozens" (10-100), or "hundreds_plus" (100+). Estimate only what is IN THE FRAME.
- protester_violence: are the protesters/civilians using force (throwing objects, fighting, setting fires, destroying property)? "none", "low" (isolated or minor), or "high" (widespread).
- state_presence: "yes" if police, riot police, soldiers, or security forces are visible; "no" otherwise.
- state_violence: are the security forces using force (batons, shields pressed into people, tear gas, water cannon, arrests by force)? "none", "low", or "high". Use "none" if no security forces are present.
- evidence: the specific visible features that justify the above -- what is actually in the pixels. Name what you see.
- confidence: "high", "medium", or "low".

Decision rules:
- Code only what is VISIBLE, not what is plausible or typical for such a scene.
- A dense crowd is not a protest by itself. Look for signs, banners, or a collective political demand.
- Keep protester_violence and state_violence separate: a clash can be one, the other, both, or neither. Attribute force to whoever is actually using it.
- If the image is too blurry or cropped to tell, say so with "unclear"/"none" and confidence "low" rather than guessing.

Respond with ONLY a JSON object, no other text:
{"is_protest": "...", "crowd_size": "...", "protester_violence": "...", "state_presence": "...", "state_violence": "...", "evidence": "...", "confidence": "..."}
"""

REQUIRED = ["is_protest", "crowd_size", "protester_violence",
            "state_presence", "state_violence", "evidence", "confidence"]
VALID = {
    "is_protest": {"yes", "no", "unclear"},
    "crowd_size": {"none", "handful", "dozens", "hundreds_plus"},
    "protester_violence": {"none", "low", "high"},
    "state_presence": {"yes", "no"},
    "state_violence": {"none", "low", "high"},
    "confidence": {"high", "medium", "low"},
}

# %%
# Run it on the peaceful march, and validate with parse_json (course_utils'
# generalization of this morning's parser: it checks the keys exist AND each
# constrained value is from our allowed set).
if HAVE_KEY:
    raw = chat_image(PROTEST_CODEBOOK, example_path)
    print(raw, "\n")
    obj = parse_json(raw, required_keys=REQUIRED, valid_values=VALID)
    print("Parsed:", obj)
else:
    print("[skipped -- no API key]")
    print('Typical output for protest1.png: is_protest "yes", crowd_size')
    print('"dozens", protester_violence "none", state_presence "no",')
    print('state_violence "none", with evidence naming the handmade signs and')
    print("the marching crowd. parse_json returns it as a dict.")

# %% [markdown]
# Read the `evidence` field against the image. Does it describe things you can
# actually see? Comparing the model's stated evidence to your own eyes is the
# move you'll keep making all afternoon, and the one you'll still be doing on
# your own images where there is no gold at all.

# %% [markdown]
# ## Step 3: Code the demo set, check the ordering
#
# Now run the codebook over all six images. We don't have exact gold, but we
# have the paper's low/high designation, which is enough to check the model's
# *ordering*: it should call `protest0` (soccer) not-a-protest and the rest
# protests; it should rate `protestor_violence1` above `protestor_violence0` on
# protester violence, and `state_violence1` above `state_violence0` on state
# violence. Ordering agreement is a weaker claim than accuracy, and it's the
# honest one to make with six images.

# %%
from tqdm.auto import tqdm

def code_images(paths, prompt):
    """Run the codebook over a list of image paths. Returns a list of dicts
    (or None where the call or the parse failed)."""
    out = []
    for p in tqdm(paths):
        try:
            raw = chat_image(prompt, p)
        except Exception as e:
            print(f"Hard failure on {os.path.basename(p)}, recording None: {e}")
            raw = None
        out.append(parse_json(raw, required_keys=REQUIRED, valid_values=VALID))
    return out

demo_paths = [os.path.join(PROTEST_DIR, f) for f in protest_files]

if HAVE_KEY:
    coded = code_images(demo_paths, PROTEST_CODEBOOK)
    rows = []
    for f, r in zip(protest_files, coded):
        row = {"filename": f}
        row.update(r if r else {k: None for k in REQUIRED})
        rows.append(row)
    demo = pd.DataFrame(rows)
    demo.to_csv(os.path.join(OUTPUTS_DIR, "lab6_protest_codes.csv"), index=False)
    print(f"Parse failures: {demo['is_protest'].isna().sum()} / {len(demo)}")
    print(demo[["filename", "is_protest", "crowd_size",
                "protester_violence", "state_violence", "confidence"]])
else:
    demo = None
    print("[skipped -- no API key] Typical run: 0 parse failures. is_protest")
    print("is 'yes' for all but protest0.png (the soccer image), which the")
    print("model usually -- not always -- calls 'no'. The violence fields")
    print("usually order correctly: the '1' images rate above the '0' images.")

# %%
# Score the ordering against PAPER_TRUTH. Three checks, each a rank comparison,
# not an exact-label comparison.
VIOL_RANK = {"none": 0, "low": 1, "high": 2}

def check_ordering(demo):
    d = demo.set_index("filename")
    # 1. is_protest: soccer should be no, marches should be yes.
    for f, t in PAPER_TRUTH.items():
        if "is_protest" in t and f in d.index:
            got = d.loc[f, "is_protest"]
            ok = "OK " if got == t["is_protest"] else ">> "
            print(f"{ok}is_protest {f}: paper={t['is_protest']}, model={got}")
    # 2. protester violence ordering.
    pv = d.loc["protestor_violence1.png", "protester_violence"], d.loc["protestor_violence0.png", "protester_violence"]
    print(f"\nprotester_violence: pv1={pv[0]} vs pv0={pv[1]} -> "
          f"{'OK, ordered' if VIOL_RANK.get(pv[0],0) > VIOL_RANK.get(pv[1],0) else '>> NOT ordered'}")
    # 3. state violence ordering.
    sv = d.loc["state_violence1.png", "state_violence"], d.loc["state_violence0.png", "state_violence"]
    print(f"state_violence:     sv1={sv[0]} vs sv0={sv[1]} -> "
          f"{'OK, ordered' if VIOL_RANK.get(sv[0],0) > VIOL_RANK.get(sv[1],0) else '>> NOT ordered'}")

if HAVE_KEY and demo is not None:
    check_ordering(demo)
else:
    print("[skipped -- no labels to score]")

# %% [markdown]
# ### Read the evidence on any disagreement
#
# Wherever the model broke the expected ordering, or you'd have coded it
# differently, open the image and read the evidence field. Three questions for
# each: was the model wrong, was the image genuinely ambiguous, or is the
# boundary *in our codebook* (what counts as "low" vs "high" violence?) doing the
# damage? The last one is the interesting case — it's a codebook problem, not a
# model problem, and you fix it by editing prose, exactly like this morning.

# %%
if HAVE_KEY and demo is not None:
    d = demo.set_index("filename")
    for f in protest_files:
        print(f"{f}  ({PAPER_TRUTH[f]['note']})")
        print(f"  model: is_protest={d.loc[f,'is_protest']}, "
              f"pv={d.loc[f,'protester_violence']}, sv={d.loc[f,'state_violence']}")
        print(f"  evidence: {d.loc[f,'evidence']}\n")
else:
    print("[skipped -- no labels]")

# %% [markdown]
# ### try it here: run the codebook on YOUR images
#
# This is the real exercise. Run the same codebook over the images you uploaded
# and read every evidence field against the actual picture. On your own images
# there's no `PAPER_TRUTH` to fall back on — your eyes are the only validation
# you have, which is exactly the situation you'll be in for a real project.

# %%
if HAVE_KEY and my_paths:
    my_coded = code_images([prepare_image(p) for p in my_paths], PROTEST_CODEBOOK)
    for p, r in zip(my_paths, my_coded):
        print(os.path.basename(p), "->", r, "\n")
    # Display one so you can grade it:
    # Image.open(my_paths[0])
elif not my_paths:
    print("No images in my_images/ yet -- go back to the upload cell.")
else:
    print("[skipped -- no API key]")

# %% [markdown]
# ## Step 4: Failure modes
#
# Everything so far made the VLM look good, because clean, well-lit, single-scene
# photos are the easy case. The next four exercises push on the places the
# chapter warned about. Each maps onto a real MMCHIVED measurement problem, and
# each is one of the reasons the paper trained dedicated models instead of asking
# a single model to describe the photo.

# %% [markdown]
# ### 4a. Counting: protest size
#
# Protest size is one of MMCHIVED's headline variables, and notice how the paper
# gets it: not by asking a model "how many people," but by running a
# face-detection model and *summing the detected faces*. That's a deliberate
# choice, and this cell shows why. Ask the VLM for a crowd count and you'll get a
# confident number that wobbles on rerun and falls apart as the crowd grows.

# %%
crowd_path = os.path.join(PROTEST_DIR, "protest1.png")
Image.open(crowd_path)

# %%
if HAVE_KEY:
    for i in range(3):
        print(f"run {i+1}:", chat_image(
            "How many people are visible in this photo? Reply with just a number.",
            crowd_path))
else:
    print("[skipped -- no API key] Typical result: three different numbers")
    print("across three runs (e.g. 40, 60, 35), none of them checkable. VLMs")
    print("estimate small counts in clean scenes okay and degrade fast with")
    print("crowd size, occlusion, and clutter -- which is exactly the regime a")
    print("protest-size variable lives in.")

# %% [markdown]
# This isn't really about the VLM being bad. A count is a measurement, and a
# number that changes when you re-ask is telling you its own error bar is large.
# If protest size is your dependent variable, a VLM estimate is a hypothesis;
# the measurement comes from a detection model (Day 2's world) or from counting
# something the model can actually enumerate, like distinct signs.

# %%
# try it here: does a VLM count YOUR images better? Pick one of your uploads
# with a countable number of something (people, vehicles, signs, buildings),
# ask three times, and see how stable the answer is. Then count it yourself.

# %% [markdown]
# ### 4b. Confabulation: describing what isn't there
#
# Confabulation is the failure that doesn't show up in your error metrics: with
# no gold to score against, a confident wrong description looks just like a
# confident right one. When you hand a VLM an image with little real information,
# it *can* pattern-complete from its training data and hand you a fluent,
# specific description of a scene that isn't in the pixels.
#
# But there's a catch, and it's why this cell is a dial and not a fixed demo: an
# image degraded to obvious mush is the *easy* case — a decent model just says
# "too blurry to tell," which is the honest answer. Confabulation lives in the
# middle, on an image that still looks interpretable but no longer supports the
# specifics. So we make the degradation adjustable. Your job is to find the level
# where this model stops abstaining and starts inventing.

# %%
from PIL import ImageFilter

def degrade(path, downto=200, blur=1.5):
    """Shrink an image to `downto` px on the long side, blow it back up, and
    add a little blur. Higher downto / lower blur = more readable. Lower downto
    / higher blur = more destroyed. The interesting behavior is in between."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = downto / max(w, h)
    small = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    out = small.resize((w, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(blur))
    p = os.path.join(OUTPUTS_DIR, "degraded_probe.png")
    out.save(p)
    return p, out

# Start moderately degraded -- readable-ish, but the details are gone.
blur_path, degraded_img = degrade(os.path.join(PROTEST_DIR, "state_violence1.png"),
                                  downto=200, blur=1.5)
degraded_img

# %%
if HAVE_KEY:
    print(chat_image("Describe in detail what is happening in this photo.", blur_path))
else:
    print("[skipped -- no API key] Two things can happen, and both are")
    print("instructive. (a) The model abstains -- 'the image is too low-quality")
    print("to describe' -- which is the honest answer and a good sign. (b) The")
    print("model confabulates -- a confident paragraph naming specific people,")
    print("signs, or smoke that the degraded pixels don't support. Dial `downto`")
    print("and `blur` until you find this model's transition between the two.")

# %% [markdown]
# Grade whatever came back against the degraded pixels: every specific noun the
# model produced that you cannot actually see is a confabulation. If the model
# abstained, don't stop — push the degradation the *other* way (raise `downto` so
# it's more readable, or lower `blur`) and find the band where it stops saying
# "too blurry" and starts filling in detail. Every model has one; the question is
# where.
#
# Then note the fix. It isn't a smarter model, it's the codebook discipline from
# Step 2. Rerun the *full codebook* (with its "unclear"/"none" paths and evidence
# field) on the degraded image and watch whether the structure gives the model an
# off-ramp a free-form "describe this" never offered.

# %%
if HAVE_KEY:
    raw = chat_image(PROTEST_CODEBOOK, blur_path)
    print(raw)
    print("\nParsed:", parse_json(raw, required_keys=REQUIRED, valid_values=VALID))
else:
    print("[skipped -- no API key] Typical output: lower confidence, more")
    print("'unclear'/'none', and an evidence field that (sometimes) admits the")
    print("blur. Structure doesn't make the model see; it gives it permission")
    print("to abstain, which free-form description never does.")

# %% [markdown]
# ### 4c. Suggestibility: the leading question
#
# Suggestibility is a direct casualty of the damage-assessment framing, and for
# OSINT work it's the failure I'd worry about most. The image below is a
# *peaceful* march — no injuries, no violence, the right answer to "how many
# injured?" is zero. We ask about it two ways and compare.

# %%
peaceful_path = os.path.join(PROTEST_DIR, "protest1.png")
Image.open(peaceful_path)

# %%
# Leading version: the question presupposes injuries exist.
if HAVE_KEY:
    print("LEADING:  ", chat_image(
        "How many injured protesters are visible in this photo?", peaceful_path))
else:
    print("[skipped -- no API key] VLMs have historically played along with the")
    print("premise -- 'approximately 2-3 individuals appear to be injured' -- on")
    print("images with no injuries at all. Better models increasingly resist the")
    print("blunt version; the point of the test is that you cannot know which")
    print("kind you have without an image where the answer is 'nothing'.")

# %%
# Neutral version: same image, no embedded premise, explicit zero-option.
if HAVE_KEY:
    print("NEUTRAL:  ", chat_image(
        "Are any people in this photo visibly injured? If yes, how many and how "
        "can you tell? If none, answer 'no visible injuries'.", peaceful_path))
else:
    print("[skipped -- no API key] Typical answer: 'no visible injuries' --")
    print("correct. Same image and same model, but changing the phrasing of the")
    print("question changed the answer, and that gap is your error, not the model's.")

# %% [markdown]
# Whether or not this particular model took the bait, the principle is survey
# methodology with higher stakes: a leading question produces a leading answer,
# and a VLM is an agreeable respondent. The danger for a pipeline is that the
# bias is *correlated* — if every image query embeds the same premise ("count the
# damaged buildings," "describe the violence"), whatever the model invents all
# pushes the same direction, and that's the kind of error that quietly moves a
# regression coefficient (Day 4's territory).
#
# The acceptance test for any image codebook is an image where the right answer
# is "nothing." If your codebook can't return "nothing" on a photo that contains
# nothing, then it's confirming the premise you built into the question rather
# than measuring anything in the image.

# %%
# try it here: write the damage/injury question the way you'd actually put it in
# a codebook -- neutral phrasing, an explicit "none" path, an evidence
# requirement -- and run it on peaceful_path AND on the blurred probe from 4b.
# Does your wording survive both traps?

# %% [markdown]
# ### 4d. Attribution: who is using force?
#
# MMCHIVED trains *two* violence classifiers, protester and state, because in a
# clash photo the two are tangled together and you have to attribute force to the
# right actor. This is the compositional question the chapter flags: right
# ingredients, wrong assignment. The image below has protesters and riot police
# in one frame. Can the VLM keep them straight?

# %%
clash_path = os.path.join(PROTEST_DIR, "state_violence1.png")
Image.open(clash_path)

# %%
if HAVE_KEY:
    print(chat_image(
        "In this photo, who is using physical force -- the protesters, the "
        "police/security forces, both, or neither? For each group, name exactly "
        "what you see them doing. Respond as JSON with keys "
        '"protesters", "police", "who_is_using_force".', clash_path))
else:
    print("[skipped -- no API key] Typical behavior: modern VLMs usually get the")
    print("actors right on a clear clash like this (helmets and shields = police,")
    print("the crowd = protesters). The failure to watch for is attribute-")
    print("swapping on messier frames -- force credited to the wrong side, or a")
    print("single 'there is violence' that collapses the distinction the paper")
    print("spent two classifiers to preserve.")

# %% [markdown]
# If it aced this one, don't over-update: a clear line of shielded police against
# a crowd is the easy case. Attribution gets hard when the actors look similar
# (plainclothes police), when the decisive action is small in the frame, or when
# the image is cropped to remove one side. Compositional and relational questions
# deserve *more* validation than whole-image labels, not less.

# %%
# try it here: run the same who-is-using-force question on protestor_violence1.png
# (masked protesters + water-cannon truck). Does the model correctly load the
# force onto the protesters there, and onto the police in the image above? Then
# try it on any of your own clash/crowd images.

# %% [markdown]
# ### 4e. Design your own trap (required)
#
# The four traps above were ours. The transferable skill is building them for
# your own pipeline, and it has three steps in a fixed order:
#
# 1. **State the truth.** Pick or make an image where you already know the
#    answer — one of your uploads you've eyeballed, a peaceful scene where the
#    answer is "nothing," or something you blur/crop/montage yourself with PIL.
#    Write the truth down *before* any API call.
# 2. **Predict the failure.** Which family — counting, confabulation,
#    suggestibility, attribution — and what exactly do you expect the model to
#    say?
# 3. **Run it** and score the answer against your stated truth.
#
# A trap the model survives is as informative as one it fails, but only if you
# wrote the prediction first. Bring your trap to the debrief.

# %%
# Your trap (fill in TRUTH and PREDICTION before running, VERDICT after):
TRAP_TRUTH = ""       # what is actually in your image
TRAP_PREDICTION = ""  # the failure you expect, and which family (4a-4d)

# trap_path = my_paths[0]              # or build one with PIL
# if HAVE_KEY:
#     print(chat_image("your question here", trap_path))

TRAP_VERDICT = ""     # one sentence: did it fail the way you predicted?

# %% [markdown]
# ## Step 5: So when do you use which instrument?
#
# Put the VLM next to how MMCHIVED actually built these variables in 2022:
#
# | | MMCHIVED's pipeline | VLM prompting (today) |
# |---|---|---|
# | Setup cost | thousands of hand-annotated images per classifier; a Bradley-Terry model for the continuous violence scores | a written codebook |
# | Adding a variable | collect labels, train another model | add a line to the prompt |
# | Marginal cost per image | ~free once trained | a fraction of a cent |
# | Output | one score per trained classifier | any fields you can describe in words, plus evidence |
# | Counting / size | a dedicated detection model | unreliable (you just watched it) |
# | Failure mode | visible misclassification | fluent confabulation |
#
# This is the same tradeoff you saw on *text* in the embeddings-plus-classifier
# labs, now on images: a trained head is cheap to run but rigid, while prompting
# is flexible but easy to fool. At a few thousand images the VLM wins on
# flexibility and costs pennies; at ten million it loses to a trained model (or
# to a small model distilled from VLM labels — this morning's scaling material).
# In between, real
# OSINT practice uses VLMs as **triage**: sift everything cheaply for the few
# images worth a person's time ("any police? any fire? readable signage?"), with
# humans doing verification and anything consequential.
#
# One standing warning from the chapter: **geolocation**. A VLM will hand you a
# confident country and city from architecture, vegetation, and signage. That's
# a fine hypothesis to check against a map, but treating it as evidence is
# malpractice: it's how a wrong claim gets into the record.

# %% [markdown]
# ## Capstone variant
#
# If your capstone has an image stream (Telegram channel photos, scraped news
# images, protest photos, satellite crops):
#
# 1. Write the codebook prompt for *your* construct — categories, decision
#    rules, a "cannot determine" path, a required `evidence` field, a JSON
#    schema. Your suggestibility test is the acceptance test: does the prompt
#    return "nothing" on an image that contains nothing?
# 2. Hand-code 20 of your own images *before* running the model (same reason as
#    this morning: you can't measure the instrument's error without gold you made
#    yourself).
# 3. Run, check ordering/agreement where you can, and read every disagreement
#    with the evidence field open.
#
# If your target is text-only, write a short paragraph for stand-up on *why* text
# suffices — what an image stream would add and what it would cost to validate.
# "We considered images and rejected them because..." is a sentence that belongs
# in a methods section.

# %% [markdown]
# ## If you finish early
#
# **1. Enforce JSON at the API level.** Instead of asking nicely for JSON, you
# can make the API refuse non-JSON output with
# `response_format={"type": "json_object"}`. One OpenRouter caveat: your request
# gets routed to one of several providers, and not all support `response_format`.
# Passing `extra_body={"provider": {"require_parameters": True}}` restricts
# routing to providers that honor every parameter you sent. Try it below. Does
# `parse_json` ever see malformed output now? And is the *schema* (your keys and
# allowed values) enforced, or just JSON-ness?

# %%
if HAVE_KEY:
    import base64
    client = course_utils.get_client()
    with open(example_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROTEST_CODEBOOK},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        temperature=0.0,
        response_format={"type": "json_object"},
        extra_body={"provider": {"require_parameters": True}},
    )
    print(resp.choices[0].message.content)
else:
    print("[skipped -- no API key] Expected: valid JSON every time. But note what")
    print("is NOT enforced -- json_object guarantees syntax, not your schema. A")
    print("made-up crowd_size value is still valid JSON, so parse_json's value")
    print("checks stay.")

# %% [markdown]
# **2. Does temperature make the codes wobble?** We've run everything at
# temperature 0. Re-code one of the ambiguous images three times at
# temperature 0.7 and compare. Nobody runs an image pipeline at 0.7, but the
# wobble you see there doesn't vanish at 0.0 — it just hides. (Day 5 complicates
# "deterministic" even at temperature 0.)

# %%
if HAVE_KEY:
    wobble_path = os.path.join(PROTEST_DIR, "protestor_violence1.png")
    for i in range(3):
        raw = chat_image(PROTEST_CODEBOOK, wobble_path, temperature=0.7)
        obj = parse_json(raw, required_keys=REQUIRED, valid_values=VALID)
        print(f"run {i+1}: pv={obj['protester_violence']}, sv={obj['state_violence']}, "
              f"size={obj['crowd_size']}" if obj else f"run {i+1}: parse failed")
else:
    print("[skipped -- no API key] Typical result: the violence/size fields")
    print("shift across runs on a genuinely ambiguous image. A code that changes")
    print("when you re-ask is reporting its own uncertainty.")
