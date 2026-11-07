# Disney+ Topic Modeling met LDA

Minor Big data & Design, week 10 (Lab).

## Wat is dit?
Een Python-script dat de **1450 Disney+-titels** (1052 films, 398 series) groepeert op thema
met **LDA (Latent Dirichlet Allocation)**. Het model gebruikt alleen de titel en beschrijving,
er zijn vooraf geen labels nodig. De opzet volgt het GeeksforGeeks-voorbeeld over LDA.

## Wat doet het script (`lda_disney.py`)?
1. Leest `disney_plus_titles.csv` en voegt titel + beschrijving samen.
2. Schoont de tekst op, haalt stopwoorden weg en brengt woorden terug tot hun basisvorm (spaCy).
3. Bouwt een bag-of-words corpus (gensim).
4. Traint een LDA-model met **20 topics** en meet de kwaliteit met de coherence-score.
5. Koppelt elke titel aan zijn belangrijkste topic.
6. Schrijft de resultaten naar Excel, CSV, grafieken en een interactieve LDAvis.

## Bestanden
| Bestand | Inhoud |
|---|---|
| `lda_disney.py` | Het volledige script |
| `disney_plus_titles.csv` | Invoerdata |
| `requirements.txt` | Benodigde Python-pakketten |
| `output/lda_results.xlsx` | Sheet *Topics* (topwoorden, aantal titels) en sheet *Titles* (elke titel met topic) |
| `output/lda_vis.html` | Interactieve LDAvis, open in je browser |
| `output/topics.csv` | Topwoorden per topic |
| `output/titles_with_topics.csv` | Elke titel met topic en zekerheid |
| `output/coherence_per_k.png` | Coherence bij 5, 10, 15, 20 en 25 topics |
| `output/topic_sizes.png` | Aantal titels per topic |

## Starten
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python lda_disney.py
```

## Belangrijkste resultaten
- Coherence (c_v): **0,59 bij 10 topics**, **0,53 bij 20 topics**. Meer topics geeft meer detail, maar minder samenhang.
- Herkenbare thema's: Mickey en vrienden, Kerst, Star Wars en ruimte, Muppets en Frozen.
- Topic 9 is het grootste (283 titels, algemene familiewoorden). Topic 14 het kleinste (9 titels).

## Aandachtspunten
- Beschrijvingen zijn kort (gemiddeld ongeveer 75 tekens), dus de topics zijn niet altijd scherp.
- Het woord "advisory" komt uit content-waarschuwingen en hoort bij de stopwoorden.
- De topicnamen zijn jouw interpretatie van de topwoorden, het model geeft alleen woorden.

## Instellingen aanpassen (bovenin het script)
- `NUM_TOPICS`: vast aantal (nu 20), of `None` voor het aantal met de hoogste coherence.
- `TOPIC_RANGE`: waarden voor de coherence-grafiek.
- `TEXT_COLUMNS`: welke kolommen als tekst dienen (bijvoorbeeld `listed_in` toevoegen).
