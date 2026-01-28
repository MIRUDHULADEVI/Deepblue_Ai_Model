import base64
import requests
import mimetypes
import re
import json
import os
import tempfile
from urllib.parse import urlparse

# =====================================================
# GLOBAL DEBUG FLAG
# =====================================================
DEBUG = True

def debug(title, data):
    if DEBUG:
        print(f"\n--- {title} ---")
        print(json.dumps(data, indent=2) if isinstance(data, (dict, list)) else data)

# =====================================================
# OCR CONFIG
# =====================================================
KOLOSAL_API_KEY = "kol_eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiM2U3YjRlYzktNTFhZC00MjZkLWFlYjktNWZmZTYzOGI0YzFlIiwia2V5X2lkIjoiMzdhZGRiYTEtYjM3NC00MzkwLTg1ODItODEyNTQ2YjE0ZGYyIiwia2V5X25hbWUiOiJvY3IiLCJlbWFpbCI6ImF2aWdobmFhLnRAZ21haWwuY29tIiwicmF0ZV9saW1pdF9ycHMiOm51bGwsIm1heF9jcmVkaXRfdXNlIjpudWxsLCJjcmVhdGVkX2F0IjoxNzY4MTk2Njk0LCJleHBpcmVzX2F0IjoxNzk5NzMyNjk0LCJpYXQiOjE3NjgxOTY2OTR9.x-T9lBVVK9VqPYXXQCjMp5g5_bCYnhU3xZV9x9lMK8I"
KOLOSAL_OCR_URL = "https://api.kolosal.ai/ocr"

# =====================================================
# IMAGE → BASE64
# =====================================================
def image_to_base64(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        # Fallback if mimetype is not detected or it's a temp file without extension
        mime_type = "image/jpeg"
        
    with open(image_path, "rb") as f:
        return f"data:{mime_type};base64," + base64.b64encode(f.read()).decode()

# =====================================================
# OCR CALL
# =====================================================
def kolosal_ocr(image_path):
    payload = {
        "auto_fix": True,
        "invoice": False,
        "language": "auto",
        "image_data": image_to_base64(image_path)
    }
    headers = {
        "Authorization": f"Bearer {KOLOSAL_API_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.post(KOLOSAL_OCR_URL, json=payload, headers=headers)
    r.raise_for_status()
    return r.json().get("extracted_text", "").lower()

# =====================================================
# LAB PARAMETERS (ALL 14 REPORTS)
# =====================================================
LAB_RANGES = {

# CBC
"hemoglobin": (12, 16),
"rbc": (4.2, 6.1),
"wbc": (4000, 11000),
"platelets": (1.5, 4.5),
"hematocrit": (36, 46),
"mcv": (80, 100),
"mch": (27, 33),
"mchc": (32, 36),
"rdw": (11, 15),
"neutrophils": (40, 75),
"lymphocytes": (20, 45),
"monocytes": (2, 10),
"eosinophils": (1, 6),
"basophils": (0, 2),

# LFT
"bilirubin_total": (0.3, 1.2),
"bilirubin_direct": (0.0, 0.3),
"bilirubin_indirect": (0.2, 0.9),
"sgot": (0, 40),
"sgpt": (0, 40),
"alp": (44, 147),
"albumin": (3.5, 5.5),
"globulin": (2.0, 3.5),

# KFT
"creatinine": (0.6, 1.3),
"urea": (15, 40),
"uric_acid": (3.4, 7.0),

# Diabetes
"fasting_glucose": (70, 100),
"postprandial_glucose": (70, 140),
"random_glucose": (70, 140),
"hba1c": (4.0, 5.6),

# Thyroid
"tsh": (0.4, 4.5),
"t3": (80, 200),
"t4": (5.0, 12.0),

# Lipid
"total_cholesterol": (0, 200),
"ldl": (0, 130),
"hdl": (40, 60),
"triglycerides": (0, 150),
"vldl": (5, 40),

# Electrolytes
"sodium": (135, 145),
"potassium": (3.5, 5.0),
"chloride": (98, 106),
"calcium": (8.5, 10.5),
"magnesium": (1.7, 2.4),
"phosphorus": (2.5, 4.5),

# Iron & Vitamins
"serum_iron": (60, 170),
"ferritin": (30, 400),
"tibc": (240, 450),
"vitamin_b12": (200, 900),
"vitamin_d": (20, 50),
"folate": (2.7, 17),

# Inflammatory
"crp": (0, 5),
"esr": (0, 20),

# Cardiac
"troponin": (0, 0.04),
"ck_mb": (0, 25),
"bnp": (0, 100),

# Coagulation
"inr": (0.8, 1.2),
"pt": (11, 13.5),
"aptt": (25, 35),

# Urine
"urine_protein": (0, 0),
"urine_sugar": (0, 0),
"urine_ketones": (0, 0),

# Hormonal
"prolactin": (4, 23),
"cortisol": (5, 25),
"testosterone": (300, 1000),
"estrogen": (30, 400)
}

# =====================================================
# NORMALIZATION
# =====================================================
def normalize(text):
    labs = {}
    for param in LAB_RANGES:
        m = re.search(rf"{param.replace('_',' ')}[^0-9]*([\d.]+)", text)
        if m:
            labs[param] = float(m.group(1))
    return labs

def status(param, value):
    low, high = LAB_RANGES[param]
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "normal"

# =====================================================
# 75 DISEASE RULES
# =====================================================
DISEASES = {

# CBC (15)
"Anemia": [("hemoglobin","low")],
"Iron Deficiency Anemia": [("hemoglobin","low"),("ferritin","low")],
"Megaloblastic Anemia": [("mcv","high"),("vitamin_b12","low")],
"Normocytic Anemia": [("hemoglobin","low"),("mcv","normal")],
"Polycythemia": [("hematocrit","high")],
"Leukocytosis": [("wbc","high")],
"Leukopenia": [("wbc","low")],
"Neutrophilia": [("neutrophils","high")],
"Neutropenia": [("neutrophils","low")],
"Lymphocytosis": [("lymphocytes","high")],
"Lymphopenia": [("lymphocytes","low")],
"Thrombocytopenia": [("platelets","low")],
"Thrombocytosis": [("platelets","high")],
"Pancytopenia Pattern": [("wbc","low"),("platelets","low"),("hemoglobin","low")],
"Eosinophilia": [("eosinophils","high")],

# LFT (12)
"Jaundice": [("bilirubin_total","high")],
"Hepatitis Pattern": [("sgpt","high"),("sgot","high")],
"Cholestasis": [("alp","high")],
"Alcoholic Liver Injury": [("sgot","high"),("sgpt","high")],
"Acute Hepatitis": [("sgpt","high")],
"Chronic Liver Disease": [("albumin","low")],
"Hypoalbuminemia": [("albumin","low")],
"Direct Hyperbilirubinemia": [("bilirubin_direct","high")],
"Indirect Hyperbilirubinemia": [("bilirubin_indirect","high")],
"Fatty Liver Risk Pattern": [("triglycerides","high")],
"Liver Synthetic Dysfunction": [("albumin","low"),("inr","high")],
"Hepatic Inflammation": [("crp","high")],

# KFT (8)
"Acute Kidney Injury": [("creatinine","high"),("urea","high")],
"Chronic Kidney Disease": [("creatinine","high"),("calcium","low")],
"Renal Impairment": [("creatinine","high")],
"Azotemia": [("urea","high")],
"Hyperuricemia": [("uric_acid","high")],
"Dehydration (Renal)": [("urea","high")],
"Reduced Renal Clearance": [("creatinine","high")],
"Renal Tubular Dysfunction": [("electrolyte","abnormal")],

# Diabetes (5)
"Hypoglycemia": [("fasting_glucose","low")],
"Prediabetes": [("fasting_glucose","high")],
"Diabetes Mellitus": [("hba1c","high")],
"Poor Glycemic Control": [("hba1c","high"),("urine_sugar","high")],
"Diabetic Nephropathy Risk": [("urine_protein","high")],

# Thyroid (6)
"Hypothyroidism": [("tsh","high")],
"Hyperthyroidism": [("tsh","low")],
"Subclinical Hypothyroidism": [("tsh","high"),("t4","normal")],
"Subclinical Hyperthyroidism": [("tsh","low"),("t4","normal")],
"Thyroid Hormone Imbalance": [("t3","abnormal")],
"Non-specific Thyroid Dysfunction": [("tsh","abnormal")],

# Lipid (6)
"Dyslipidemia": [("total_cholesterol","high")],
"Atherogenic Lipid Pattern": [("ldl","high"),("hdl","low")],
"Hypertriglyceridemia": [("triglycerides","high")],
"Low HDL Risk": [("hdl","low")],
"Mixed Hyperlipidemia": [("ldl","high"),("triglycerides","high")],
"Cardiovascular Lipid Risk": [("ldl","high")],

# Electrolytes (7)
"Hyponatremia": [("sodium","low")],
"Hypernatremia": [("sodium","high")],
"Hypokalemia": [("potassium","low")],
"Hyperkalemia": [("potassium","high")],
"Hypocalcemia": [("calcium","low")],
"Hypercalcemia": [("calcium","high")],
"Electrolyte Imbalance Syndrome": [("sodium","abnormal")],

# Vitamins / Iron (6)
"Iron Deficiency": [("ferritin","low")],
"Iron Overload": [("ferritin","high")],
"Vitamin B12 Deficiency": [("vitamin_b12","low")],
"Vitamin D Deficiency": [("vitamin_d","low")],
"Folate Deficiency": [("folate","low")],
"Nutritional Deficiency Pattern": [("vitamin_d","low"),("vitamin_b12","low")],

# Inflammatory (4)
"Acute Inflammation": [("crp","high")],
"Chronic Inflammation": [("esr","high")],
"Systemic Inflammatory Response": [("crp","high"),("wbc","high")],
"Possible Infection / Sepsis Risk": [("crp","high"),("neutrophils","high")],

# Cardiac (3)
"Myocardial Injury": [("troponin","high")],
"Cardiac Stress Pattern": [("ck_mb","high")],
"Heart Failure Risk": [("bnp","high")],

# Coagulation / Urine (3)
"Bleeding Risk": [("inr","high")],
"Proteinuria": [("urine_protein","high")],
"Glycosuria": [("urine_sugar","high")]
}

# =====================================================
# DISEASE INFERENCE
# =====================================================
def infer_diseases(labs):
    results = []

    for disease, rules in DISEASES.items():
        matched = 0
        evidence = []

        for param, expected in rules:
            if param not in labs:
                break

            s = status(param, labs[param])

            if expected == "abnormal":
                if s == "normal":
                    break
                evidence.append(f"{param} is {s}")
                matched += 1
            else:
                if s != expected:
                    break
                evidence.append(f"{param} is {s}")
                matched += 1

        if matched == len(rules):
            results.append({
                "disease": disease,
                "confidence": 100.0,
                "evidence": evidence
            })

    if not results:
        return [{
            "disease": "No significant abnormality detected",
            "confidence": 95.0,
            "evidence": []
        }]
    
    # Sort to ensure highest confidence or priority is first if needed
    # (Though logic above returns 100% matches)
    return results

# =====================================================
# BRIDGE FUNCTION FOR APP.PY
# =====================================================
def predict_ocr(message: str):
    """
    Main entry point for app.py to call.
    Accepts an image path, URL, or Base64 Data URI.
    Returns a dict with 'text' and 'analysis'.
    """
    try:
        input_str = message.strip()
        temp_file = None
        image_path = None

        # 1. Check for Data URI (Base64)
        if input_str.startswith("data:"):
            try:
                print(f"[OCR] Processing Data URI...")
                header, encoded = input_str.split(",", 1)
                # content-type is inside header: data:image/png;base64
                data = base64.b64decode(encoded)
                
                # Guess extension
                ext = ".jpg"
                if "image/png" in header: ext = ".png"
                elif "image/jpeg" in header: ext = ".jpg"
                elif "image/webp" in header: ext = ".webp"
                
                temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
                with os.fdopen(temp_fd, 'wb') as f:
                    f.write(data)
                
                image_path = temp_path
                temp_file = temp_path
            except Exception as e:
                return {
                    "text": "Error processing image data.",
                    "analysis": f"Could not decode base64 image: {str(e)}"
                }

        # 2. Check if input is a URL (http/https)
        elif input_str.startswith("http"):
            try:
                print(f"[OCR] Downloading image from URL: {input_str}")
                parsed = urlparse(input_str)
                response = requests.get(input_str, stream=True)
                response.raise_for_status()
                
                # Create temp file
                suffix = os.path.splitext(parsed.path)[1] or ".jpg"
                temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(temp_fd, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                image_path = temp_path
                temp_file = temp_path 
            except Exception as e:
                return {
                    "text": "Error downloading image.",
                    "analysis": f"Could not download image from URL: {str(e)}"
                }
        else:
            # 3. Assume local path
            possible_path = input_str.strip('"').strip("'")
            if os.path.exists(possible_path):
                 image_path = possible_path
            else:
                 # If it's not a path, maybe it's just text or invalid
                 return {
                    "text": "Image not found.",
                    "analysis": f"Could not find image at path: {possible_path}. Please upload an image or provide a valid URL."
                }

        # 4. Perform OCR & Analysis
        print(f"[OCR] Processing image: {image_path}")
        extracted_text = kolosal_ocr(image_path)
        labs = normalize(extracted_text)
        diagnoses = infer_diseases(labs)
        
        # 5. Build Analysis String (mimicking the user's print logic)
        top = diagnoses[0]
        
        analysis_lines = []
        analysis_lines.append("POSSIBLE CONDITIONS:")
        for d in diagnoses:
            analysis_lines.append(f"• {d['disease']}")
            
        analysis_lines.append(f"\n🔥 MOST LIKELY CONDITION:")
        if top['disease'] == "No significant abnormality detected":
             analysis_lines.append(f"👉 {top['disease']}")
        else:
             analysis_lines.append(f"👉 {top['disease']} ({top['confidence']}%)")
        
        if top["evidence"]:
            analysis_lines.append("\nReason:")
            for e in top["evidence"]:
                analysis_lines.append(f"• {e}")
        
        analysis_text = "\n".join(analysis_lines)

        # Cleanup temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

        return {
            "text": extracted_text,
            "analysis": analysis_text,
            "predicted_disease": top['disease']
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "text": "Error processing OCR.",
            "analysis": f"An error occurred during analysis: {str(e)}"
        }

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    # Example usage for testing
    print("This module is intended to be imported. Run app.py instead.")
