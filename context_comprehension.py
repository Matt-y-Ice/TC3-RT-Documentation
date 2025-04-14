import spacy
import re

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Global variable for confirmed placement
CONFIRMED_PLACEMENT = None

def extract_injury_details(text):
    """
    Extracts injuries (laterality + body part).
    """
    injury_data = []
    pattern = re.compile(r'\b(?:(left|right)\s+)?(arm|leg|forearm|thigh|knee)\b', re.IGNORECASE)
    for laterality, part in pattern.findall(text):
        part = part.lower()
        ext = "arm" if part in {"arm", "forearm"} else "leg"
        injury_data.append({
            "laterality": laterality.lower() if laterality else None,
            "extremity": ext,
            "location_phrase": (f"{laterality} {part}" if laterality else part).strip()
        })
    return injury_data

def extract_tourniquet_details(text):
    """
    Detects tourniquet mentions and looks across the full transcript for matching anatomy.
    """
    global CONFIRMED_PLACEMENT
    doc = nlp(text)

    suggestion_keywords = {"should", "could", "would", "suggest", "maybe", "might", "recommend", "consider"}
    tourniquet_sentences = []
    tourniquet_data = []

    # First pass: find all sentences mentioning tourniquet
    for sent in doc.sents:
        if any("tourniquet" in token.lemma_.lower() for token in sent):
            sent_text = sent.text.strip()
            status = "possible tourniquet placement" if any(
                token.lemma_.lower() in suggestion_keywords for token in sent
            ) else "confirmed tourniquet placement"

            tourniquet_sentences.append((sent_text, status))

    # Second pass: search the entire transcript for anatomy context
    anatomy_mentions = []
    for token in doc:
        word = token.text.lower()
        if word in {"left", "right"}:
            laterality = word
            next_token = token.nbor(1) if token.i + 1 < len(doc) else None
            if next_token and next_token.text.lower() in {"arm", "forearm", "leg", "thigh", "knee"}:
                part = next_token.text.lower()
                extremity = "arm" if "arm" in part else "leg"
                anatomy_mentions.append({
                    "laterality": laterality,
                    "extremity": extremity,
                    "location_phrase": f"{laterality} {part}"
                })

    # Match each tourniquet sentence with best available context
    for sent_text, status in tourniquet_sentences:
        matched = anatomy_mentions[0] if anatomy_mentions else {"laterality": None, "extremity": None, "location_phrase": "unspecified"}

        if status == "confirmed tourniquet placement":
            CONFIRMED_PLACEMENT = f"Confirmed tourniquet placement at {matched['location_phrase']}"

        tourniquet_data.append({
            "status": status,
            "laterality": matched["laterality"],
            "extremity": matched["extremity"],
            "location_phrase": matched["location_phrase"],
            "sentence": sent_text
        })

    return tourniquet_data

def assemble_prediction_string(injury_data, tourniquet_data):
    """
    Builds the final detection string for reporting.
    """
    predictions = []

    for injury in injury_data:
        predictions.append(f"Injury at {injury['location_phrase']}")

    for tq in tourniquet_data:
        if tq['status'] == "confirmed tourniquet placement":
            predictions.append(f"Confirmed tourniquet placement at {tq['location_phrase']}")
        else:
            predictions.append(f"Suggested/Questioned tourniquet placement at {tq['location_phrase']}")

    if not tourniquet_data:
        predictions.append("No tourniquet placement detected")

    return " | ".join(predictions)

def process_transcript(text):
    """
    Runs the full pipeline on a single transcript string.
    """
    global CONFIRMED_PLACEMENT
    CONFIRMED_PLACEMENT = None

    injury_data = extract_injury_details(text)
    tourniquet_data = extract_tourniquet_details(text)
    print("Tourniquet Data:", tourniquet_data)
    prediction_string = assemble_prediction_string(injury_data, tourniquet_data)

    result = {
        "injuries": injury_data,
        "tourniquets": tourniquet_data,
        "prediction_string": prediction_string,
        "confirmed_global": CONFIRMED_PLACEMENT
    }

    CONFIRMED_PLACEMENT = None  # Reset after processing
    return result