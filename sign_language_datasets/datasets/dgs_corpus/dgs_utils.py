import pympi


def get_elan_sentences(elan_path: str):
    eaf = pympi.Elan.Eaf(elan_path)  # TODO add "suppress_version_warning=True" when pympi 1.7 is released

    timeslots = eaf.timeslots

    for participant in ["A", "B"]:
        german_tier_name = "Deutsche_Übersetzung_" + participant
        if german_tier_name not in eaf.tiers:
            continue

        german_text = list(eaf.tiers[german_tier_name][0].values())

        english_tier_name = "Translation_into_English_" + participant
        english_text = list(eaf.tiers[english_tier_name][0].values()) if english_tier_name in eaf.tiers else []

        all_glosses = []
        for hand in ["r", "l"]:
            hand_tier = "Lexem_Gebärde_" + hand + "_" + participant
            if hand_tier not in eaf.tiers:
                continue

            gloss = {
                _id: {"start": timeslots[s], "end": timeslots[e], "gloss": val, "hand": hand}
                for _id, (s, e, val, _) in eaf.tiers[hand_tier][0].items()
            }
            for tier in ["Lexeme_Sign", "Gebärde", "Sign"]:
                items = eaf.tiers[tier + "_" + hand + "_" + participant][1]
                for ref, val, _1, _2 in items.values():
                    if ref in gloss:  # 2 files have a missing reference
                        gloss[ref][tier] = val

            all_glosses += list(gloss.values())

        for (s, e, val, _) in german_text:
            sentence = {"participant": participant, "start": timeslots[s], "end": timeslots[e], "german": val}

            # Add English sentence
            english_sentence = [val for (s2, e2, val2, _) in english_text if s == s2 and e == e2]
            sentence["english"] = english_sentence[0] if len(english_sentence) > 0 else None

            # Add glosses
            sentence["glosses"] = list(
                sorted(
                    [item for item in all_glosses if item["start"] >= sentence["start"] and item["end"] <= sentence["end"]],
                    key=lambda d: d["start"],
                )
            )

            yield sentence
