import pympi

"""
Notes:

Annotation guidelines of third release: https://archive.mpi.nl/tla/islandora/object/tla%3A1839_00_0000_0000_0021_6AC3_1

Gloss annotations:
- Glosses are annotated separately for each hand
- the gloss tiers are called "GlossL" and "GlossR"
- two-handed signs are annotated identically on the tiers of both hands (this also inflates the overall count of
  annotated glosses

The translations into Dutch are in the tier called "TranslationNarrow".

Both gloss and translation tiers exist for both speakers S1 and S2. Hence, the actual tier names of interest are:
- GlossL S1
- GlossL S2
- GlossR S1
- GlossR S2
- TranslationNarrow S1
- TranslationNarrow S2

Not all ELAN files have gloss annotations or translations. The code needs to assume that any tier type could be missing.
"""


def get_elan_sentences_ngt_corpus(elan_path: str):

    eaf = pympi.Elan.Eaf(elan_path)  # TODO add "suppress_version_warning=True" when pympi 1.7 is released

    timeslots = eaf.timeslots

    for participant in ["S1", "S2"]:
        dutch_tier_name = "TranslationNarrow " + participant
        if dutch_tier_name not in eaf.tiers:
            continue

        # tiers is defined as follows (http://dopefishh.github.io/pympi/Elan.html):
        #   {tier_name -> (aligned_annotations, reference_annotations, attributes, ordinal)}
        # aligned_annotations:
        #   [{id -> (begin_ts, end_ts, value, svg_ref)}]
        # reference_annotations:
        #   [{id -> (reference, value, previous, svg_ref)}]
        #
        # - "ts" means timeslot, which references a time value in miliseconds
        # - "value" is the actual annotation content
        # - "svg_ref" is an optional reference to an SVG image that is always None in our files

        dutch_text = list(eaf.tiers[dutch_tier_name][0].values())

        # collect all glosses in the entire ELAN file

        all_glosses = []

        for hand in ["R", "L"]:
            hand_tier = "Gloss" + hand + " " + participant
            if hand_tier not in eaf.tiers:
                continue

            glosses = {}

            for gloss_id, (start, end, value, _) in eaf.tiers[hand_tier][0].items():
                glosses[gloss_id] = {"start": timeslots[start],
                                     "end": timeslots[end],
                                     "gloss": value,
                                     "hand": hand}

            all_glosses += list(glosses.values())

        for (start, end, value, _) in dutch_text:
            sentence = {"participant": participant,
                        "start": timeslots[start],
                        "end": timeslots[end],
                        "dutch": value}

            # Add glosses whose timestamps are within this sentence
            glosses_in_sentence = [item for item in all_glosses if
                                   item["start"] >= sentence["start"]
                                   and item["end"] <= sentence["end"]]

            sentence["glosses"] = list(sorted(glosses_in_sentence, key=lambda d: d["start"]))

            yield sentence
