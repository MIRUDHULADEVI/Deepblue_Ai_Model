import json
import os

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_CURRENT_DIR)
_DATA_DIR = os.path.join(_BACKEND_DIR, "data")

def get_disease_info(disease_name):
    # Load data
    try:
        with open(os.path.join(_DATA_DIR, "disease.json"), encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"error": "Disease database not found."}

    # Search for disease by name
    for entry in data:
        if entry.get("disease_name", "").lower() == disease_name.lower():
            return {
                "name": entry["disease_name"],
                "definition": entry.get("definitions", "Definition not available."),
                "symptoms": entry.get("symptoms", []),
                "causes": entry.get("causes", [])
            }
            
    return {"error": "Disease not found."}
