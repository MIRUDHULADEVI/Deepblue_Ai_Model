from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# -------------------------------------------------------------
import os
# Load fine-tuned Navarasa model
# -------------------------------------------------------------
# Get the directory of the current file (backend/models/symptom_model.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to 'backend'
backend_dir = os.path.dirname(current_dir)
model_path = os.path.join(backend_dir, "navarasa-symptom-checker-final")

base_model_name = "Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa-2.0"
# Global variables for lazy loading
_tokenizer = None
_model = None

def get_model():
    global _tokenizer, _model

    if _model is None:
        print("[INFO] Loading Navarasa model (lazy)...")

        try:
            _tokenizer = AutoTokenizer.from_pretrained(base_model_name)

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,   # ✅ CPU-safe
                low_cpu_mem_usage=True       # ✅ critical
            )

            _model = PeftModel.from_pretrained(base_model, model_path)
            _model.eval()

            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token

            print("[INFO] Model loaded successfully")
        
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return None, None

    return _tokenizer, _model


# -------------------------------------------------------------
# Language Detection (10 supported languages)
# -------------------------------------------------------------
def detect_language(text):
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

from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# -------------------------------------------------------------
# Supported Languages (Navarasa Model)
# -------------------------------------------------------------
SUPPORTED_LANGUAGES = {'ta', 'mr', 'hi', 'te', 'ml', 'kn', 'gu', 'as', 'bn', 'en'}

# -------------------------------------------------------------
# Disease Map (English – MASTER)
# -------------------------------------------------------------
disease_map_en = {
    0: "Anemia", 1: "Diabetes", 2: "Hypertension", 3: "Tuberculosis",
    4: "Chikungunya", 5: "Dengue", 6: "Typhoid", 7: "Diarrhea",
    8: "Malnutrition", 9: "Cholera", 10: "Dehydration",
    11: "Age-Related Macular Degeneration", 12: "Conjunctivitis (Pink Eye)",
    13: "Asthma", 14: "Influenza (Flu)", 15: "Pneumonia", 16: "Constipation",
    17: "Sinusitis", 18: "Food Poisoning", 19: "Fungal Infections",
    20: "Fatty Liver", 21: "Arthritis", 22: "Heat Stroke",
    23: "Hepatitis B", 24: "Hepatitis C", 25: "Liver Cirrhosis",
    26: "Alzheimer's Disease", 27: "Gastritis", 28: "Peptic Ulcer",
    29: "Parkinson's Disease", 30: "Breast Cancer", 31: "Cervical Cancer",
    32: "Urinary Tract Infection", 33: "Lupus", 34: "Scurvy",
    35: "Ringworm", 36: "Nipah Virus Infection", 37: "Zika Virus Disease",
    38: "Yellow Fever", 39: "Polycystic Ovary Syndrome",
    40: "Hypothyroidism", 41: "Hyperthyroidism", 42: "COPD",
    43: "Endometriosis", 44: "Cataract", 45: "HIV/AIDS",
    46: "Oral Cancer", 47: "Stomach Cancer", 48: "Liver Cancer",
    49: "Lung Cancer", 50: "Piles", 51: "Eczema", 52: "Acne", 53: "Jaundice",
    54: "Iron Deficiency Anemia", 55: "Megaloblastic Anemia", 56: "Normocytic Anemia",
    57: "Polycythemia", 58: "Leukocytosis", 59: "Leukopenia",
    60: "Neutrophilia", 61: "Neutropenia", 62: "Lymphocytosis",
    63: "Lymphopenia", 64: "Thrombocytopenia", 65: "Thrombocytosis",
    66: "Pancytopenia Pattern", 67: "Eosinophilia", 68: "Hepatitis Pattern",
    69: "Cholestatis", 70: "Alcoholic Liver Injury", 71: "Acute Hepatitis",
    72: "Chronic Liver Disease", 73: "Hypoalbuminemia", 74: "Direct Hyperbilirubinemia",
    75: "Indirect Hyperbilirubinemia", 76: "Fatty Liver Risk Pattern", 77: "Liver Synthetic Dysfunction",
    78: "Hepatic Inflammation", 79: "Acute Kidney Injury", 80: "Chronic Kidney Disease",
    81: "Renal Impairment", 82: "Azotemia", 83: "Hyperuricemia",
    84: "Dehydration (Renal)", 85: "Reduced renal Clearance", 86: "Renal Tubular Dysfunction",
    87: "Hypoglycemia", 88: "Prediabetes", 89: "Diabetes Mellitus",
    90: "Poor Glycemic control", 91: "Diabetic Nephropathy Risk", 92: "Subclinical Hypothyroidism",
    93: "Subclinical Hyperthyroidism", 94: "Thyroid Hormone Imbalance", 95: "Non-Specific Thyroid Dysfunction",
    96: "Dyslipidemia", 97: "Atherogenic Lipid Pattern", 98: "Hypertriglyceridemia",
    99: "Low HDL Risk", 100: "Mixed Hyperlipidemia", 101: "Cardiovascular Lipid Risk",
    102: "Hyponatremia", 103: "Hypernatremia", 104: "Hypokalemia",
    105: "Hyperkalemia", 106: "Hypocalcemia", 107: "Hypercalcemia",
    108: "Electrolyte Imbalance Syndrome", 109: "Iron Deficiency", 110: "Iron Overload",
    111: "Vitamin B12 Deficiency", 112: "Vitamin D Deficiency", 113: "Folate Deficiency",
    114: "Nutritional Deficiency Pattern", 115: "Acute Inflammation", 116: "Chronic Inflammation",
    117: "Systemic Inflammatory Response", 118: "Possible Infection Risk", 119: "Myocardial Injury",
    120: "Cardiac Stress Pattern", 121: "Heart Failure Risk", 122: "Bleeding Risk",
    123: "Proteinuria", 124: "Glycosuria"
}

disease_map = {lang: disease_map_en for lang in ["en", "ta", "hi", "te", "ml", "kn", "mr", "gu", "as", "bn"]}

# -------------------------------------------------------------
# Urgency Map
# -------------------------------------------------------------
urgency_map = {
    "en": {0: "Self-care", 1: "Doctor Visit", 2: "Emergency"},
    "hi": {0: "स्व-देखभाल", 1: "डॉक्टर से मिलें", 2: "आपातकाल"},
    "ta": {0: "சுய பராமரிப்பு", 1: "மருத்துவரை சந்திக்கவும்", 2: "அவசரம்"},
    "te": {0: "స్వీయ సంరక్షణ", 1: "డాక్టర్ను కలవండి", 2: "అత్యవసరం"},
    "ml": {0: "സ്വയം പരിചരണം", 1: "ഡോക്ടറെ കാണുക", 2: "ആപത്ത്"},
    "kn": {0: "ಸ್ವಯಂ ಆರೈಕೆ", 1: "ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ", 2: "ತುರ್ತು"},
    "gu": {0: "સ્વ-કાળજી", 1: "ડોક્ટરને મળો", 2: "એમર્જન્સી"},
    "mr": {0: "स्वतः काळजी", 1: "डॉक्टरांना भेटा", 2: "आपत्काल"},
    "bn": {0: "স্ব-যত্ন", 1: "ডাক্তারের কাছে যান", 2: "জরুরী"},
    "as": {0: "নিজকে যত্ন লওক", 1: "ডাক্টৰৰ ওচৰলৈ যাওক", 2: "জৰুৰী অৱস্থা"}
}

# -------------------------------------------------------------
# Generation-based Prediction
# -------------------------------------------------------------
def predict_disease_urgency(text):
    # 1. Detect Language
    try:
        detected_lang = detect(text)
    except LangDetectException:
        detected_lang = 'en'
    
    # Check if manual override logic (unicode based) is needed or if langdetect is sufficient.
    # We will trust langdetect mostly, but fallback to 'en' if unsure.
    # Also, we explicitly support the 10 languages.
    
    processing_text = text
    processing_lang = detected_lang

    if detected_lang not in SUPPORTED_LANGUAGES:
        print(f"[-] Language '{detected_lang}' not directly supported. Translating to English...")
        try:
            processing_text = GoogleTranslator(source='auto', target='en').translate(text)
            processing_lang = 'en'
        except Exception as e:
            print(f"[!] Translation failed: {e}. Proceeding with original text.")
            processing_lang = 'en' # Fallback assumption

    # Format prompt for Navarasa instruction tuning
    prompt = f"""### Instruction:
Based on these symptoms, predict the disease and urgency level (0=low, 1=medium, 2=high).

### Input:
Symptoms: {processing_text}

### Response:
Disease: """

    try:
        tokenizer, model = get_model()
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return processing_lang, 22, 2

    if model is None:
        print("[WARNING] Model not loaded (returned None), returning mock prediction")
        return processing_lang, 22, 2 # Heat Stroke fallback

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,  # Low temperature for consistent predictions
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract numbers from response (Disease: X, Urgency: Y)
    import re
    disease_match = re.search(r'Disease:\s*(\d+)', full_response)
    urgency_match = re.search(r'Urgency:\s*(\d+)', full_response)

    disease_id = int(disease_match.group(1)) if disease_match else 0
    urgency_id = int(urgency_match.group(1)) if urgency_match else 0

    return processing_lang, disease_id, urgency_id

def predict_multilingual(text):
    lang, disease_id, urgency_id = predict_disease_urgency(text)

    # Map disease/urgency using the processing language (which matches our Maps)
    # If the original input was French -> detected 'fr' -> translated to 'en' -> lang='en'.
    # So we use disease_map['en'], which is correct.
    
    disease_name = disease_map.get(lang, disease_map["en"]).get(
        disease_id, f"Unknown Disease ({disease_id})"
    )
    urgency_name = urgency_map.get(lang, urgency_map["en"]).get(
        urgency_id, f"Unknown Urgency ({urgency_id})"
    )

    return {
        "language": lang,
        "disease_id": disease_id,
        "disease": disease_name,
        "urgency_id": urgency_id,
        "urgency": urgency_name
    }

# -------------------------------------------------------------
# Test
# -------------------------------------------------------------
if __name__ == '__main__':
    user_input = "for several weeks fatigue, weakness, pale skin, mild dizziness, headache, brittle nails, cold hands and feet, restless legs"
    print(f"Input: {user_input}")
    print(predict_multilingual(user_input))

    # Test Tamil input
    tamil_input = "மென்மையான தலைவலி, மயக்கம், வாந்தி, அதிக வியர்வை, பலவீனம்"
    print(f"\nInput: {tamil_input}")
    print(predict_multilingual(tamil_input))
