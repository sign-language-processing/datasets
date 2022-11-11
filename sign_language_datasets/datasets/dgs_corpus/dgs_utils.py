import pympi

from lxml import etree
from typing import Dict, List


def get_elan_sentences(elan_path: str):

    eaf = pympi.Elan.Eaf(elan_path)  # TODO add "suppress_version_warning=True" when pympi 1.7 is released

    timeslots = eaf.timeslots

    for participant in ["A", "B"]:
        german_tier_name = "Deutsche_Übersetzung_" + participant
        if german_tier_name not in eaf.tiers:
            continue

        german_text = eaf.tiers[german_tier_name][0]

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

        all_mouthings = []

        tier_name = "Mundbild_Mundgestik_" + participant
        items = eaf.tiers[tier_name][0]

        # structure of entries:
        # {'a2768296': ('ts42', 'ts43', 'tochter', None), ... }

        for s, e, val, _ in items.values():
            mouthing_entry = {"start": timeslots[s], "end": timeslots[e], "mouthing": val}
            all_mouthings.append(mouthing_entry)

        for _id, (s, e, val, _) in german_text.items():
            sentence = {"id": _id, "participant": participant, "start": timeslots[s], "end": timeslots[e], "german": val}

            # Add English sentence
            english_sentence = [val2 for (s2, e2, val2, _) in english_text if s == s2 and e == e2]
            sentence["english"] = english_sentence[0] if len(english_sentence) > 0 else None

            # Add glosses
            sentence["glosses"] = list(
                sorted(
                    [item for item in all_glosses if item["start"] >= sentence["start"] and item["end"] <= sentence["end"]],
                    key=lambda d: d["start"],
                )
            )

            # add mouthings
            sentence["mouthings"] = list(
                sorted(
                    [item for item in all_mouthings if item["start"] >= sentence["start"] and item["end"] <= sentence["end"]],
                    key=lambda d: d["start"],
                )
            )

            yield sentence


def get_child_elements(root: etree.ElementTree,
                       element_name: str,
                       attributes_to_extract: List[str]) -> Dict[str, Dict[str, str]]:
    """

    :param root:
    :param element_name:
    :param attributes_to_extract:
    :return:
    """

    elements = root.xpath("/ilex-data/" + element_name)  # type: List[etree.Element]

    by_id = {}

    for element in elements:
        id_ = element.get("id")
        by_id[id_] = {}
        for attribute_name in attributes_to_extract:
            value = element.get(attribute_name)
            by_id[id_][attribute_name] = value

    return by_id


def get_signer_ids_from_ilex(ilex_path: str) -> Dict[str, List[str]]:
    """

    File structure:

    <ilex-data source="meinedgs.de" version="1.1" database_version="51">
        <camera_perspective id="1" code="A1" english="Frontal view on informant A"
            localised="Frontalansicht Informant A" visible_persons="{1}"/>
        <camera_perspective id="2" code="B1" english="Frontal view on informant B"
            localised="Frontalansicht Informant B" visible_persons="{2}"/>
        <camera_perspective id="3" code="C" english="Total on all three persons"
            localised="Totale auf alle drei Personen" visible_persons="{2,3,1}"/>
        <movie_track id="3" movie="1" camera_perspective="3" path="./1177918_1c.mp4"
            track_length="00:09:25:04"/>
        <movie_track id="1" movie="1" camera_perspective="1" path="./1177918_1a1.mp4"
            track_length="00:09:25:04"/>
        <movie_track id="2" movie="1" camera_perspective="2" path="./1177918_1b1.mp4"
            track_length="00:09:25:04"/>
        <informant id="1" sex="1" name="SH-12" short_name="SH-12"/>
        <informant id="2" sex="1" name="SH-13" short_name="SH-13"/>
        <informant id="3" sex="2" name="sh-mod-1" short_name="sh-mod-1"/>
        <participation id="1" movie="1" role="1" informant="1"/>
        <participation id="2" movie="1" role="1" informant="2"/>
        <participation id="3" movie="1" role="2" informant="3"/>
        <!--...-->
    </ilex-data>

    :param ilex_path:
    :return:
    """

    root = etree.parse(ilex_path)

    informant_dict = get_child_elements(root=root,
                                        element_name="informant",
                                        attributes_to_extract=["name"])

    camera_perspective_dict = get_child_elements(root=root,
                                                 element_name="camera_perspective",
                                                 attributes_to_extract=["visible_persons", "code"])

    signer_identities_by_perspective = {}  # type: Dict[str, List[str]]

    for camera_perspective in camera_perspective_dict.values():

        # extract A, B or C without trailing numbers

        clean_code = camera_perspective["code"][0].lower()

        # remove enclosing "{" and "}" for list of informant ids

        ids_of_visible_persons = camera_perspective["visible_persons"][1:-1].split(",")

        names_of_visible_persons = [informant_dict[id_]["name"] for id_ in ids_of_visible_persons]

        signer_identities_by_perspective[clean_code] = names_of_visible_persons

    return signer_identities_by_perspective
