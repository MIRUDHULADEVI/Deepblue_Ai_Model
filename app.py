import json
import os
from typing import TypedDict, List
from flask import Flask, request, jsonify
from flask_cors import CORS
from langgraph.graph import StateGraph, END

from models.symptom_model import predict_multilingual
from models.ocr_model import predict_ocr
from models.diet_model import get_diet_advice
from models.disease_info_model import get_disease_info

# ------------------------------------------------- 
# Flask App
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------
# Load Medical Data
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

with open(os.path.join(DATA_DIR, "disease.json"), encoding="utf-8") as f:
    raw_disease_list = json.load(f)

# Re-index DISEASE_DB by disease_label and language
DISEASE_DB = {}
for entry in raw_disease_list:
    did = str(entry.get("disease_label", ""))
    lang = entry.get("lang", "en")
    if did:
        DISEASE_DB.setdefault(did, {})[lang] = entry

# Create Reverse Mapping (Name -> ID) for OCR lookup
DISEASE_NAME_TO_ID = {}
for entry in raw_disease_list:
    name = entry.get("disease_name", "").lower()
    did = entry.get("disease_label")
    if name and did is not None:
        DISEASE_NAME_TO_ID[name] = str(did)
    # Also map aliases
    for alias in entry.get("aliases", []):
        DISEASE_NAME_TO_ID[alias.lower()] = str(did)

with open(os.path.join(DATA_DIR, "recommendation.json"), encoding="utf-8") as f:
    rec_list = json.load(f)

PRESCRIPTION_DB = {}
for item in rec_list:
    did = str(item.get("disease_id", ""))
    lang = item.get("lang", "en")
    if did:
        PRESCRIPTION_DB.setdefault(did, {})[lang] = item.get("recommendation", {})

# -------------------------------------------------
# LangGraph State
# -------------------------------------------------
class AgentState(TypedDict):
    message: str
    selected_option: str
    step: str
    response: str
    options: List[str]
    disease_id: str
    urgency: str
    language: str
    viewed_sections: List[str]  # Track which sections have been viewed



# -------------------------------------------------
# Translation Templates
# -------------------------------------------------
# -------------------------------------------------
# Translation Templates
# -------------------------------------------------
TRANSLATIONS = {
    "en": {
        "identified": "Based on your symptoms, we have identified: {disease}",
        "urgency_label": "Urgency Level: {urgency}",
        "ask_next": "What would you like to know more about?",
        "disease_label": "Disease:",
        "def_label": "Definition:",
        "sym_label": "Common Symptoms:",
        "rec_header": "Recommendations:",
        "do_label": "Do:",
        "dont_label": "Don't:",
        "remedy_label": "Home Remedies:",
        "no_data": "No recommendation data available.",
        "error_model": "Error: Model returned no result. Please try again.",
        "desc_opt": "Disease Description",
        "rec_opt": "Diet Recommendation",
        "start_opt": "Start Over",
        "no_sym": "No symptoms listed"
    },
    "ta": {
        "identified": "உங்கள் அறிகுறிகளின் அடிப்படையில், நாங்கள் கண்டறிந்தது: {disease}",
        "urgency_label": "அவசர நிலை: {urgency}",
        "ask_next": "நீங்கள் எதைப் பற்றி மேலும் அறிய விரும்புகிறீர்கள்?",
        "disease_label": "நோய்:",
        "def_label": "விளக்கம்:",
        "sym_label": "பொதுவான அறிகுறிகள்:",
        "rec_header": "பரிந்துரைகள்:",
        "do_label": "செய்ய வேண்டியவை:",
        "dont_label": "செய்யக்கூடாதவை:",
        "remedy_label": "வீட்டு வைத்தியம்:",
        "no_data": "பரிந்துரை தரவு கிடைக்கவில்லை.",
        "error_model": "பிழை: மாதிரி எந்த முடிவும் அளிக்காவிட்டால் மீண்டும் முயற்சிக்கவும்.",
        "desc_opt": "Disease Description (நோய் விளக்கம்)",
        "rec_opt": "Diet Recommendation (உணவு பரிந்துரை)",
        "start_opt": "Start Over (மீண்டும் தொடங்க)",
        "no_sym": "அறிகுறிகள் எதுவும் பட்டியலிடப்படவில்லை"
    },
    "hi": {
        "identified": "आपके लक्षणों के आधार पर, हमने पहचाना है: {disease}",
        "urgency_label": "तात्कालिकता स्तर: {urgency}",
        "ask_next": "आप किसके बारे में अधिक जानना चाहेंगे?",
        "disease_label": "रोग:",
        "def_label": "परिभाषा:",
        "sym_label": "सामान्य लक्षण:",
        "rec_header": "सिफारिशें:",
        "do_label": "क्या करें:",
        "dont_label": "क्या न करें:",
        "remedy_label": "घरेलू उपचार:",
        "no_data": "कोई सिफारिश डेटा उपलब्ध नहीं है।",
        "error_model": "त्रुटि: मॉडल ने कोई परिणाम नहीं दिया। कृपया पुनः प्रयास करें।",
        "desc_opt": "Disease Description (रोग विवरण)",
        "rec_opt": "Diet Recommendation (आहार सिफारिश)",
        "start_opt": "Start Over (शुरू से करें)",
        "no_sym": "कोई लक्षण सूचीबद्ध नहीं"
    }
}

def get_text(lang, key, **kwargs):
    """Helper to get translated text safe, falling back to English"""
    t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    text = t.get(key, TRANSLATIONS["en"].get(key, ""))
    if kwargs:
        return text.format(**kwargs)
    return text


# -------------------------------------------------
# Dispatcher Node (Entry Point)
# -------------------------------------------------
def dispatcher_node(state: AgentState):
    """Routes to the appropriate node based on current step"""
    print(f"[DISPATCHER] step={state.get('step')}, option={state.get('selected_option')}")
    # Just pass through - the router will handle the actual routing
    return {}


def start_node(state: AgentState):
    """Shows the initial menu"""
    print(f"[START_NODE] Showing menu")
    return {
        "response": "Please select a service:",
        "options": [
            "OPTION_1: Symptom Checker",
            "OPTION_2: Scan Report (OCR)",
            "OPTION_3: Disease Information",
            "OPTION_4: Diet Advice"
        ],
        "step": "start",
        "selected_option": ""
    }


def symptom_checker_node(state: AgentState):
    print(f"[SYMPTOM_CHECKER_NODE] Processing message: {state['message'][:50]}...")
    
    msg = state.get("message", "").strip()
    option = state.get("selected_option", "")

    
    # If message is empty or just the option text (meaning no user input yet), ask for input
    if not msg or msg == option or msg == "OPTION_1":
        print("[SYMPTOM_CHECKER_NODE] No symptoms provided, asking user.")
        return {
            "response": "Please describe your symptoms in detail ",
            "options": [],
            "step": "symptom_input",
            "selected_option": option,
            "disease_id": "",
            "urgency": "",
            "language": state.get("language", "en")
        }

    try:
        result = predict_multilingual(msg)
        print(f"[SYMPTOM_CHECKER_NODE] Model result: {result}")
        
        if result is None:
            return {
                "response": get_text("en", "error_model"),
                "options": ["Start Over"],
                "step": "start"
            }
        
        detected_lang = result.get("language", "en")
        did = str(result.get("disease_id", ""))
        
        # Get localized disease name from DB if available
        db_entry = DISEASE_DB.get(did, {}).get(detected_lang) or DISEASE_DB.get(did, {}).get("en")
        disease_name = db_entry.get("disease_name", result.get("disease", "Unknown")) if db_entry else result.get("disease", "Unknown")
        
        urgency = result.get("urgency", "Unknown")

        res_text = (
            f"{get_text(detected_lang, 'identified', disease=disease_name)}\n\n"
            f"{get_text(detected_lang, 'urgency_label', urgency=urgency)}\n\n"
            f"{get_text(detected_lang, 'ask_next')}"
        )
        
        opts = [
            get_text(detected_lang, "desc_opt"),
            get_text(detected_lang, "rec_opt"),
            get_text(detected_lang, "start_opt")
        ]

        return {
            "response": res_text,
            "options": opts,
            "disease_id": did,
            "urgency": urgency,
            "language": detected_lang,
            "step": "symptom_result"
        }
    except Exception as e:
        print(f"[SYMPTOM_CHECKER_NODE] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "response": f"Error processing symptoms: {str(e)}",
            "options": ["Start Over"],
            "step": "start"
        }


def disease_description_node(state: AgentState):
    print(f"[DISEASE_DESCRIPTION_NODE] disease_id={state.get('disease_id')}, lang={state.get('language')}")
    did = str(state.get("disease_id", ""))
    lang = state.get("language", "en")
    
    # Get translation-specific details
    disease_entry = DISEASE_DB.get(did, {}).get(lang) or DISEASE_DB.get(did, {}).get("en", {})
    disease_name = disease_entry.get("disease_name", "Unknown")
    definition = disease_entry.get("definitions", "Definition not available.")
    symptoms = disease_entry.get("symptoms", [])
    
    symptoms_text = ", ".join(symptoms) if symptoms else get_text(lang, "no_sym")
    
    res_text = (
        f"{get_text(lang, 'disease_label')} {disease_name}\n\n"
        f"{get_text(lang, 'def_label')} {definition}\n\n"
        f"{get_text(lang, 'sym_label')} {symptoms_text}"
    )
    
    # Track that description has been viewed
    viewed = state.get("viewed_sections", [])
    if "description" not in viewed:
        viewed = viewed + ["description"]
    
    # Only show Diet Recommendation if it hasn't been viewed yet
    if "recommendation" in viewed:
        # Both have been viewed, only show Start Over
        opts = [get_text(lang, "start_opt")]
    else:
        # Show remaining option and Start Over
        opts = [get_text(lang, "rec_opt"), get_text(lang, "start_opt")]

    return {
        "response": res_text,
        "options": opts,
        "disease_id": did,
        "urgency": state.get("urgency", ""),
        "language": lang,
        "step": "disease_description",
        "viewed_sections": viewed
    }


def recommendation_node(state: AgentState):
    print(f"[RECOMMENDATION_NODE] disease_id={state.get('disease_id')}, lang={state.get('language')}")
    did = str(state.get("disease_id", ""))
    lang = state.get("language", "en")

    pres_lang_map = PRESCRIPTION_DB.get(did, {})
    pres = pres_lang_map.get(lang) or pres_lang_map.get("en")

    if not pres:
        return {
            "response": get_text(lang, "no_data"),
            "options": [get_text(lang, "start_opt")],
            "step": "start",
            "selected_option": ""
        }

    response = get_text(lang, "rec_header") + "\n"

    if pres.get("do"):
        response += f"\n{get_text(lang, 'do_label')}\n" + "\n".join(f"- {x}" for x in pres["do"])

    if pres.get("dont"):
        response += f"\n\n{get_text(lang, 'dont_label')}\n" + "\n".join(f"- {x}" for x in pres["dont"])

    if pres.get("home_remedies"):
        response += f"\n\n{get_text(lang, 'remedy_label')}\n" + "\n".join(f"- {x}" for x in pres["home_remedies"])

    # Track that recommendation has been viewed
    viewed = state.get("viewed_sections", [])
    if "recommendation" not in viewed:
        viewed = viewed + ["recommendation"]
    
    # Only show Disease Description if it hasn't been viewed yet
    if "description" in viewed:
        # Both have been viewed, only show Start Over
        opts = [get_text(lang, "start_opt")]
    else:
        # Show remaining option and Start Over
        opts = [get_text(lang, "desc_opt"), get_text(lang, "start_opt")]

    return {
        "response": response,
        "options": opts,
        "step": "recommendation",
        "selected_option": "",
        "viewed_sections": viewed
    }


def disease_info_node(state: AgentState):
    print(f"[DISEASE_INFO_NODE] Processing: {state['message']}")
    info = get_disease_info(state["message"])

    if "error" in info:
        return {
            "response": info["error"],
            "options": ["Start Over"],
            "step": "start",
            "selected_option": ""
        }

    return {
        "response": (
            f"Disease: {info['name']}\n\n"
            f"Definition: {info['definition']}\n\n"
            f"Symptoms: {', '.join(info['symptoms'])}"
        ),
        "options": ["Start Over"],
        "step": "start",
        "selected_option": ""
    }


def diet_node(state: AgentState):
    print(f"[DIET_NODE] Processing query")
    
    # Check if we have a disease_id from previous steps
    disease_id = state.get("disease_id")
    msg = state.get("message", "").strip()
    
    # If no ID, try to use the message as the disease name
    query = disease_id if disease_id else msg
    
    # If using message, clean it up (remove "OPTION_4: " if present)
    if isinstance(query, str) and "OPTION_4" in query:
        # User just clicked the option but didn't provide input yet?
        # Ideally frontend handles this, but here let's be safe.
        # If msg is just "OPTION_4: Diet Advice", we need to ask for input.
        if "Diet Advice" in query and len(query) < 30: # Heuristic
             return {
                "response": "Please enter the disease name to get diet advice.",
                "options": [],
                "step": "start", # Or a new step 'diet_input' if we want loop
                "selected_option": "OPTION_4" 
            }
    
    # Use detected language if available, otherwise let get_diet_advice detect/default
    lang = state.get("language") 
    
    advice = get_diet_advice(query, language=lang)

    if "error" in advice:
        return {
            "response": advice["error"],
            "options": ["Start Over"],
            "step": "start",
            "selected_option": ""
        }

    response = "Diet Advice:\n"

    if advice.get("do"):
        response += "\nEat:\n" + "\n".join(f"- {x}" for x in advice["do"])

    if advice.get("dont"):
        response += "\n\nAvoid:\n" + "\n".join(f"- {x}" for x in advice["dont"])
        
    if advice.get("home_remedies"):
         response += "\n\nHome Remedies:\n" + "\n".join(f"- {x}" for x in advice["home_remedies"])

    return {
        "response": response,
        "options": ["Start Over"],
        "step": "start",
        "selected_option": ""
    }


def ocr_node(state: AgentState):
    print(f"[OCR_NODE] Processing OCR request")
    result = predict_ocr(state["message"])

    # Extract Predicted Disease
    disease_name = result.get("predicted_disease", "")
    disease_id = DISEASE_NAME_TO_ID.get(disease_name.lower())

    response_text = (
        f"OCR Text:\n{result.get('text', '')}\n\n"
        f"Interpretation:\n{result.get('analysis', '')}"
    )

    # If we identified a valid disease ID, offer further options
    if disease_id:
        print(f"[OCR_NODE] Identified disease ID: {disease_id} for '{disease_name}'")
        # Use English prompts for now as OCR is primarily English-based
        opts = [
            get_text("en", "desc_opt"),
            get_text("en", "rec_opt"),
            get_text("en", "start_opt")
        ]
        return {
            "response": response_text,
            "options": opts,
            "disease_id": disease_id,
            "language": "en", # Default to English for OCR results for now
            "step": "ocr_result",
            "selected_option": ""
        }
    else:
        # No specific disease identified or mapped
        return {
            "response": response_text,
            "options": ["Start Over"],
            "step": "start",
            "selected_option": ""
        }


# -------------------------------------------------
# Router
# -------------------------------------------------
def router(state: AgentState):
    step = state.get("step", "start")
    option = state.get("selected_option", "")
    
    print(f"[ROUTER] step={step}, option={option}")

    if "Start Over" in option:
        return "start"

    # From dispatcher: route based on step and option
    if step == "start":
        if not option:
            # Show menu
            print("[ROUTER] Routing to: start (show menu)")
            return "start"
        else:
            # Route to selected service
            if "OPTION_1" in option: return "symptom_checker"
            if "OPTION_2" in option: return "ocr"
            if "OPTION_3" in option: return "disease_info"
            if "OPTION_4" in option: return "diet"
            
            print(f"[ROUTER] Unknown option at start: {option}, going to END")
            return END

    # From symptom_input logic
    if step == "symptom_input":
        msg = state.get("message", "").strip()
        option = state.get("selected_option", "")
        # If we have meaningful input (not just the start option), go to checker
        if msg and msg != option and "OPTION_1" not in msg:
            print("[ROUTER] Input received -> returning to symptom_checker")
            return "symptom_checker"
        else:
            print("[ROUTER] Waiting for user input -> END")
            return END

    # From symptom_checker to disease_description
    if step == "symptom_result" and "Disease Description" in option:
        print("[ROUTER] Routing to: disease_description")
        return "disease_description"
    
    # From symptom_checker to recommendations
    if step == "symptom_result" and "Diet Recommendation" in option:
        print("[ROUTER] Routing to: recommendation")
        return "recommendation"

    # From OCR Result to Disease Description
    if step == "ocr_result" and "Disease Description" in option:
        print("[ROUTER] OCR -> disease_description")
        return "disease_description"

    # From OCR Result to Recommendations
    if step == "ocr_result" and "Diet Recommendation" in option:
        print("[ROUTER] OCR -> recommendation")
        return "recommendation"

    # CROSS NAVIGATION: From disease_description to recommendations
    if step == "disease_description" and "Diet Recommendation" in option:
        print("[ROUTER] Routing to: recommendation")
        return "recommendation"

    # CROSS NAVIGATION: From recommendation to disease_description
    if step == "recommendation" and "Disease Description" in option:
        print("[ROUTER] Routing to: disease_description")
        return "disease_description"
    
    # Default end if no valid option matched
    if step in ["symptom_result", "disease_description", "recommendation"] and not option:
        print(f"[ROUTER] At {step} step with no option, ending")
        return END

    # Default: end the flow
    print("[ROUTER] Default END")
    return END


# -------------------------------------------------
# LangGraph
# -------------------------------------------------
graph = StateGraph(AgentState)

# Add all nodes
graph.add_node("dispatcher", dispatcher_node)
graph.add_node("start", start_node)
graph.add_node("symptom_checker", symptom_checker_node)
graph.add_node("disease_description", disease_description_node)
graph.add_node("recommendation", recommendation_node)
graph.add_node("disease_info", disease_info_node)
graph.add_node("diet", diet_node)
graph.add_node("ocr", ocr_node)

# Set dispatcher as entry point
graph.set_entry_point("dispatcher")

# From dispatcher, route based on state
graph.add_conditional_edges("dispatcher", router)

# From start (menu), end (user will make a new request with option)
graph.add_edge("start", END)

# From symptom_checker, can go to disease_description, recommendation, or end
graph.add_conditional_edges("symptom_checker", router)

# From disease_description, can go to recommendation (cross nav) or end
graph.add_conditional_edges("disease_description", router)

# From recommendation, can go to disease_description (cross nav) or end
graph.add_conditional_edges("recommendation", router)

# From OCR (if disease found), can go to description/recommendation
graph.add_conditional_edges("ocr", router)

# From other services, they end directly
graph.add_edge("disease_info", END)
graph.add_edge("diet", END)

graph = graph.compile()

# -------------------------------------------------
# API Routes
# -------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Backend running", "endpoints": ["/chat"]})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}

    state: AgentState = {
        "message": data.get("message", ""),
        "selected_option": data.get("selected_option", "") or data.get("option", ""),
        "step": data.get("step", "start"),
        "response": "",
        "options": [],
        "disease_id": data.get("disease_id", ""),
        "urgency": data.get("urgency", ""),
        "language": data.get("language", "en"),
        "viewed_sections": data.get("viewed_sections", [])
    }

    try:
        print(f"[-] Processing step: {state['step']}, option: {state['selected_option']}")
        result = graph.invoke(state)
        
        # Ensure result has all necessary fields for frontend
        final_result = {**state, **(result if isinstance(result, dict) else {})}
        
        print(f"[+] Result step: {final_result.get('step')}")
        return jsonify(final_result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            **state,
            "response": f"Error in backend: {str(e)}",
            "options": ["Start Over"],
            "step": "start"
        })


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
