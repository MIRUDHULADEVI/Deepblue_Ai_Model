import json
import os

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_CURRENT_DIR)
_DATA_DIR = os.path.join(_BACKEND_DIR, "data")

def detect_language(text):
    """
    Detects the language of the input text based on unicode ranges.
    Supports: ta, mr, hi, te, ml, kn, gu, as, bn, en.
    """
    if not text or not isinstance(text, str):
        return "en"
        
    for ch in text:
        code = ord(ch)
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"   # Tamil
        elif 0x0900 <= code <= 0x097F:
            return "mr" if "ळ" in text else "hi"  # Marathi / Hindi
        elif 0x0C00 <= code <= 0x0C7F:
            return "te"   # Telugu
        elif 0x0D00 <= code <= 0x0D7F:
            return "ml"   # Malayalam
        elif 0x0C80 <= code <= 0x0CFF:
            return "kn"   # Kannada
        elif 0x0A80 <= code <= 0x0AFF:
            return "gu"   # Gujarati
        elif 0x0980 <= code <= 0x09FF:
            return "as" if "ৰ" in text else "bn"  # Assamese / Bengali
    return "en"  # English fallback

def get_diet_advice(disease_name_or_id, language=None):
    # Load data
    try:
        with open(os.path.join(_DATA_DIR, "recommendation.json"), encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"error": "Diet database not found."}

    # If the user provides a string input, we try to detect the language from it.
    if language is None:
        if isinstance(disease_name_or_id, str) and not disease_name_or_id.isdigit():
            language = detect_language(disease_name_or_id)
        else:
            language = "en"
            
    disease_id = None
    
    # Try to resolve disease_id if input is string
    if isinstance(disease_name_or_id, str):
        if disease_name_or_id.isdigit():
             disease_id = int(disease_name_or_id)
        else:
            try:
                with open(os.path.join(_DATA_DIR, "disease.json"), encoding="utf-8") as f:
                    d_list = json.load(f)
                    input_processed = disease_name_or_id.strip() # maintain case for some langs, but usually lower is better for en
                    input_lower = input_processed.lower() 
                    
                    found = False
                    for entry in d_list:
                        # Check name (case-insensitive)
                        if entry.get("disease_name", "").lower() == input_lower:
                            disease_id = entry.get("disease_label")
                            found = True
                            break
                        
                        # Check aliases
                        aliases = [a.lower() for a in entry.get("aliases", [])]
                        if input_lower in aliases:
                            disease_id = entry.get("disease_label")
                            found = True
                            break
                    
                    # If not found with exact match, maybe try to match any entry regardless of language?
                    # The above loop iterates ALL entries in disease.json which includes all languages.
                    # So if input is Tamil "சர்க்கரை நோய்", it should match the Tamil entry aliases.
            except:
                pass
    else:
        disease_id = disease_name_or_id

    if disease_id is None:
         return {"error": f"Disease '{disease_name_or_id}' not found. Please try a different name."}

    # Find the recommendation for the ID and Language
    rec = None
    target_id_str = str(disease_id)
    
    # First pass: try to find exact match for ID and Language
    for item in data:
        if str(item.get("disease_id")) == target_id_str:
            if item.get("lang") == language:
                rec = item
                break
    
    # Second pass: Fallback to English if the specific language is not found
    if not rec:
        for item in data:
             if str(item.get("disease_id")) == target_id_str and item.get("lang") == "en":
                rec = item
                break

    if rec:
        return rec["recommendation"]
    else:
        return {"error": "No diet advice found for this disease."}
