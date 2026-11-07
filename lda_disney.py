"""
Topic modeling (LDA) op Disney+ titels.
Gebaseerd op het GeeksforGeeks-voorbeeld:
https://www.geeksforgeeks.org/nlp/topic-modeling-using-latent-dirichlet-allocation-lda/

Stappen: laden -> opschonen -> tokenizen + stopwoorden -> lemmatizen
         -> dictionary + corpus -> LDA -> coherence -> clusters opslaan
"""
import re
from pathlib import Path

import gensim
import gensim.corpora as corpora
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
from gensim.models import CoherenceModel
from nltk.corpus import stopwords

# ---------------------------------------------------------------- instellingen
BASE = Path(__file__).parent
CSV_PATH = BASE / "disney_plus_titles.csv"
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

TEXT_COLUMNS = ["title", "description"]  # kolommen die samen de "tekst" vormen
NUM_TOPICS = 20                           # vast aantal; None = beste aantal automatisch zoeken
TOPIC_RANGE = [5, 10, 15, 20, 25]         # waarden voor de coherence-grafiek
RANDOM_STATE = 100

# ------------------------------------------------------------------ 1. laden
df = pd.read_csv(CSV_PATH)
df["text"] = df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)

# ----------------------------------------------------------- 2. preprocessing
try:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
except LookupError:  # geen internet / download geblokkeerd -> gensim-lijst
    from gensim.parsing.preprocessing import STOPWORDS
    stop_words = set(STOPWORDS)
# woorden die in vrijwel elke Disney+-beschrijving voorkomen en niets zeggen
stop_words |= {"disney", "series", "film", "movie", "show", "season", "one",
               "new", "find", "must", "get", "take", "make", "come", "go"}


def preprocess_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\S*@\S*\s?", "", text)   # e-mails
    text = re.sub(r"'", "", text)            # apostrofs
    text = re.sub(r"[^a-zA-Z]", " ", text)   # niet-letters
    return text.lower()


def tokenize(text: str) -> list[str]:
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    return [t for t in tokens if t not in stop_words]


df["cleaned_text"] = df["text"].apply(preprocess_text)
df["tokens"] = df["cleaned_text"].apply(tokenize)

# ------------------------------------------------------------ 3. lemmatiseren
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lemmatize(tokens: list[str]) -> list[str]:
    doc = nlp(" ".join(tokens))
    return [t.lemma_ for t in doc
            if t.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}
            and t.lemma_ not in stop_words and len(t.lemma_) > 2]


df["lemmas"] = df["tokens"].apply(lemmatize)

# ------------------------------------------------- 4. dictionary en corpus
id2word = corpora.Dictionary(df["lemmas"])
id2word.filter_extremes(no_below=3, no_above=0.5)  # zeldzame/te vaak woorden eruit
texts = df["lemmas"].tolist()
corpus = [id2word.doc2bow(t) for t in texts]


# --------------------------------------------------------------- 5. LDA model
def build_lda(k: int):
    return gensim.models.ldamodel.LdaModel(
        corpus=corpus, id2word=id2word, num_topics=k,
        random_state=RANDOM_STATE, update_every=1, chunksize=100,
        passes=10, alpha="auto", per_word_topics=True,
    )


def coherence(model) -> float:
    return CoherenceModel(model=model, texts=texts, dictionary=id2word,
                          coherence="c_v").get_coherence()


scores = {}
for k in TOPIC_RANGE:
    scores[k] = coherence(build_lda(k))
    print(f"num_topics={k:2d}  coherence={scores[k]:.4f}")
if NUM_TOPICS is None:
    NUM_TOPICS = max(scores, key=scores.get)
    print(f"\nBeste aantal topics: {NUM_TOPICS}")

plt.figure(figsize=(7, 4))
plt.plot(list(scores), list(scores.values()), marker="o")
plt.axvline(NUM_TOPICS, color="grey", linestyle="--", label=f"gekozen: {NUM_TOPICS}")
plt.xlabel("Aantal topics")
plt.ylabel("Coherence (c_v)")
plt.title("Coherence per aantal topics")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "coherence_per_k.png", dpi=150)
plt.close()

lda_model = build_lda(NUM_TOPICS)

# --------------------------------------------------------------- 6. topics
print("\nTopics:")
topic_rows = []
for tid, words in lda_model.show_topics(num_topics=NUM_TOPICS, num_words=10,
                                        formatted=False):
    word_list = ", ".join(w for w, _ in words)
    print(f"Topic {tid}: {word_list}")
    topic_rows.append({"topic": tid, "top_words": word_list})
pd.DataFrame(topic_rows).to_csv(OUT / "topics.csv", index=False)

print("\nCoherence score:", round(coherence(lda_model), 4))

# ------------------------------------------ 7. elke titel aan een cluster koppelen
dominant, prob = [], []
for bow in corpus:
    dist = lda_model.get_document_topics(bow, minimum_probability=0)
    t, p = max(dist, key=lambda x: x[1])
    dominant.append(t)
    prob.append(round(float(p), 4))

df["topic"] = dominant
df["topic_probability"] = prob
cols = ["show_id", "type", "title", "listed_in", "description",
        "topic", "topic_probability"]
df[cols].to_csv(OUT / "titles_with_topics.csv", index=False)

# grootte van elk cluster
sizes = df["topic"].value_counts().sort_index()
plt.figure(figsize=(7, 4))
plt.bar(sizes.index.astype(str), sizes.values)
plt.xlabel("Topic")
plt.ylabel("Aantal titels")
plt.title("Aantal titels per topic")
plt.tight_layout()
plt.savefig(OUT / "topic_sizes.png", dpi=150)
plt.close()

# ------------------------------------------------------ 8. Excel-bestand
topics_df = pd.DataFrame(topic_rows)
topics_df["aantal_titels"] = topics_df["topic"].map(sizes).fillna(0).astype(int)
with pd.ExcelWriter(OUT / "lda_results.xlsx", engine="openpyxl") as xw:
    topics_df.to_excel(xw, sheet_name="Topics", index=False)
    df[cols].sort_values(["topic", "topic_probability"], ascending=[True, False]) \
        .to_excel(xw, sheet_name="Titles", index=False)
    for ws in xw.book.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 70)
print("Excel opgeslagen:", OUT / "lda_results.xlsx")

# ------------------------------------------- 9. interactieve visualisatie (LDAvis)
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

vis = gensimvis.prepare(lda_model, corpus, id2word, mds="mmds", sort_topics=False)
pyLDAvis.save_html(vis, str(OUT / "lda_vis.html"))
print("LDAvis opgeslagen:", OUT / "lda_vis.html")

print("\nTitels per topic:\n", sizes.to_string())
print(f"\nKlaar. Resultaten staan in: {OUT}")
