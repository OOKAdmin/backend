# --------------------------
# COMBINED FLASK APP: Beam Deflection + Plagiarism Checker
# --------------------------

import os
from dotenv import load_dotenv
load_dotenv()
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import bcrypt
import jwt
import re
from datetime import datetime
from collections import defaultdict
from pymongo import MongoClient
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
#--------------------------
# AI Detector and NLTK Initialization
#--------------------------
_nltk_ready = False  # Guard: only download NLTK data once per process

def download_nltk_resources():
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk
    resources = [
        'punkt', 'punkt_tab', 'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4', 'vader_lexicon'
    ]
    for res in resources:
        try:
            nltk.data.find(res)
        except Exception:
            nltk.download(res, quiet=True)
    _nltk_ready = True

print("Initializing AI components via Google Gemini API...")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def call_gemini(prompt, system_instruction=None):
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY is not configured on the server."
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction)
        response = model.generate_content(prompt)
        return response.text, None
    except Exception as e:
        return None, str(e)

def detector_classifier(text):
    if not GEMINI_API_KEY:
        print("[Warning] GEMINI_API_KEY is missing! AI detection is running in stub mode.")
        return [{"label": "Human", "score": 1.0}]

    prompt = f"Analyze the following text and determine if it was written by an AI or a Human. Return your answer as a raw JSON array with a single object containing 'label' and 'score'. Example: [{{\"label\": \"Fake\", \"score\": 0.99}}]. Use 'Fake' for AI and 'Human' for human. Return ONLY JSON.\n\nText: {text}"
    result_text, error = call_gemini(prompt, system_instruction="You are a JSON-only API. You output raw JSON arrays.")
    
    if error:
        print(f"[Error] Gemini API Error: {error}")
        return [{"label": "Human", "score": 1.0}]
        
    try:
        # Clean up Markdown blocks if Gemini returns them
        cleanedText = result_text.replace('```json', '').replace('```', '').strip()
        data = json.loads(cleanedText)
        if isinstance(data, list) and len(data) > 0:
            return data
        return [{"label": "Human", "score": 1.0}]
    except Exception as e:
        print(f"[Error] Parsing Gemini Response: {e} - Response was: {result_text}")
        return [{"label": "Human", "score": 1.0}]

print("[Success] AI Detector (Gemini) initialized!")

def analyze_text_ai(text):
    if not text or not text.strip() or not detector_classifier:
        return None
    
    # Analyze the whole block
    full_result = detector_classifier(text)[0]
    
    if full_result['label'] in ['Fake', 'ChatGPT']:
        ai_prob = full_result['score'] * 100
    else:
        ai_prob = (1.0 - full_result['score']) * 100
        
    final_ai_score = max(0, min(100, int(ai_prob)))

    # Exact Mood classification
    if final_ai_score >= 80:
        mood = "AI Based"
    elif final_ai_score >= 60:
        mood = "AI Based & AI Refined"
    elif final_ai_score >= 40:
        mood = "Human Written & AI Refined"
    else:
        mood = "Human Written"
        
    import nltk
    download_nltk_resources()
    sentences = nltk.sent_tokenize(text)
    segments = []
    
    # Sentence-level exact ML analysis
    for s in sentences:
        if len(s.split()) < 3:
            s_score = final_ai_score # Too short for ML, defer to average
        else:
            s_result = detector_classifier(s)[0]
            if s_result['label'] in ['Fake', 'ChatGPT']:
                s_score = s_result['score'] * 100
            else:
                s_score = (1.0 - s_result['score']) * 100
            
        if s_score >= 75:
            sType = "ai"
        elif s_score >= 55:
            sType = "ai-refined"
        elif s_score >= 35:
            sType = "human-ai"
        else:
            sType = "human"
            
        segments.append({
            "text": s.strip(),
            "type": sType
        })
        
    return {
        "aiPercentage": final_ai_score,
        "mood": mood,
        "segments": segments
    }


# Humanizer integration is now lazy-loaded inside the route.
humanizer = None

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv('API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

# ======================
# CONFIG - MongoDB / Auth
# ======================
MONGO_URI = os.getenv("MONGO_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["ook_db"]
    users = db["users"]
    # Check connection
    client.server_info()
    print("MongoDB connected successfully!")
except Exception as e:
    print(f"MongoDB connection error: {e}")
    users = None

# --------------------------
# Track plagiarism usage per IP
# --------------------------
plagiarism_trials = defaultdict(lambda: {"count": 0, "date": datetime.today().date()})

# --------------------------
# Track AI usage per IP
# --------------------------
ai_trials = defaultdict(lambda: {"count": 0, "date": datetime.today().date()})

def check_ai_limit():
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            # If the token is successfully decoded, they are authenticated. Skip limits.
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            return True, None 
        except:
            pass # Invalid token, fall through to IP limit
    
    ip = request.remote_addr
    today = datetime.today().date()
    # Reset limit if a new day
    if ai_trials[ip]["date"] < today:
        ai_trials[ip] = {"count": 0, "date": today}
    
    if ai_trials[ip]["count"] >= 8:
        return False, jsonify({"success": False, "error": "FREE_LIMIT_REACHED", "message": "You have reached your free daily limit of 8 uses. Please Log In or Sign Up to continue."})
    
    ai_trials[ip]["count"] += 1
    return True, None


# ======================
# REGISTER
# ======================
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json

    if users.find_one({"email": data["email"]}):
        return jsonify({"success": False, "error": "User already exists"})

    hashed_pw = bcrypt.hashpw(data["password"].encode(), bcrypt.gensalt())

    users.insert_one({
        "name": data["name"],
        "email": data["email"],
        "password": hashed_pw,
        "google": False
    })

    return jsonify({"success": True})


# ======================
# LOGIN
# ======================
@app.route("/api/login", methods=["POST"])
def login():
    data = request.json

    user = users.find_one({"email": data["email"]})

    if not user:
        return jsonify({"success": False, "error": "User not found"})

    if user.get("google") and not user.get("password"):
        return jsonify({"success": False, "error": "This account is linked with Google. Please use 'Sign in with Google' instead."})

    if not bcrypt.checkpw(data["password"].encode(), user["password"]):
        return jsonify({"success": False, "error": "Incorrect password"})

    token = jwt.encode({"email": user["email"]}, JWT_SECRET, algorithm="HS256")

    return jsonify({
        "success": True,
        "token": token,
        "user": {
            "name": user["name"],
            "email": user["email"],
            "photo": user.get("photo", "")
        }
    })


# ======================
# GOOGLE AUTH
# ======================
@app.route("/api/auth/google", methods=["POST"])
def google_auth():
    data = request.json

    try:
        idinfo = id_token.verify_oauth2_token(
            data["token"],
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        email = idinfo["email"]
        name = idinfo.get("name", "User")
        photo = idinfo.get("picture", "")
        user = users.find_one({"email": email})

        if not user:
            users.insert_one({
                "name": name,
                "email": email,
                "password": None,
                "google": True
            })

        token = jwt.encode({"email": email}, JWT_SECRET, algorithm="HS256")

        return jsonify({
            "success": True,
            "token": token,
            "user": {
                "name": name,
                "email": email,
                "photo": photo
            }
        })

    except Exception as e:
        print(e)
        return jsonify({"success": False, "error": "Google auth failed"})


# ======================
# FORGOT PASSWORD
# ======================
@app.route("/api/forgot-password", methods=["POST"])
def forgot_password():
    data = request.json
    email = data.get("email", "").strip().lower()

    if not email:
        return jsonify({"success": False, "error": "Email is required"})

    try:
        user = users.find_one({"email": {"$regex": f"^{email}$", "$options": "i"}})

        # Always return success to prevent user enumeration
        if user:
            # TODO: Integrate SMTP / SendGrid here to send a real reset link
            # e.g., send_reset_email(email, generate_reset_token(email))
            print(f"[ForgotPassword] Reset requested for: {email} (user found)")
        else:
            print(f"[ForgotPassword] Reset requested for: {email} (user NOT found — silent)")

        return jsonify({"success": True})

    except Exception as e:
        print(f"[ForgotPassword] Error: {e}")
        return jsonify({"success": False, "error": "Server error. Please try again later."})


# ==================================================
# PART 1: BEAM DEFLECTION API
# ==================================================
@app.route('/beam', methods=['POST'])
def handle_beam_deflection():
    length = request.json.get('length')
    point_loads = request.json.get('point_loads', [])
    distributed_loads = request.json.get('distributed_loads', [])
    supports = request.json.get('supports', [])
    youngmodules = request.json.get('youngmodules')
    area = request.json.get('area')
    inertia = request.json.get('inertia')

    import numpy as np
    try:
        from indeterminatebeam import Beam, Support, PointLoadV, TrapezoidalLoad
    except Exception as e:
        print(f"Error loading indeterminatebeam library: {e}")
        return jsonify({"error": "Beam calculation engine is currently unavailable (missing dependencies)."}), 503

    beam = Beam(length, A=area, I=inertia, E=youngmodules)

    for support in supports:
        position = support['position']
        support_type = support['type']
        if support_type == 'pinned':
            constraints = (1, 1, 0)
        elif support_type == 'roller':
            constraints = (0, 1, 0)
        elif support_type == 'fixed':
            constraints = (1, 1, 1)
        beam.add_supports(Support(position, constraints))

    for pl in point_loads:
        beam.add_loads(PointLoadV(pl['magnitude'] * -1, pl['position']))

    for dl in distributed_loads:
        beam.add_loads(TrapezoidalLoad(
            force=(dl['start_magnitude'] * -1, dl['end_magnitude'] * -1),
            span=(dl['start_position'], dl['end_position']),
            angle=90
        ))

    beam.analyse()

    reactions = []
    for support in supports:
        position = support['position']
        reaction_force = beam.get_reaction(x_coord=position)[1]
        reaction_momentum = beam.get_reaction(x_coord=position)[2]
        reactions.append({
            'position': position,
            'force': reaction_force,
            'momentum': reaction_momentum
        })

    BeamDiagram = []

    for pl in point_loads:
        BeamDiagram.append({
            'type': 'PointLoad',
            'magnitude': pl['magnitude'],
            'position': pl['position']
        })

    for dl in distributed_loads:
        BeamDiagram.append({
            'type': 'DistributedLoad',
            'start_magnitude': dl['start_magnitude'],
            'end_magnitude': dl['end_magnitude'],
            'start_position': dl['start_position'],
            'end_position': dl['end_position']
        })

    for support in supports:
        position = support['position']
        support_type = support['type']
        if support_type == 'pinned':
            constraints = (1, 1, 0)
        elif support_type == 'roller':
            constraints = (0, 1, 0)
        elif support_type == 'fixed':
            constraints = (1, 1, 1)
        BeamDiagram.append({
            'type': 'Support',
            'position': position,
            'constraints': constraints,
            'support_type': support_type
        })

    positions = np.linspace(0, length, 100)
    deflections = [beam.get_deflection(x) for x in positions]
    shear_forces = [beam.get_shear_force(x) for x in positions]
    bending_moments = [beam.get_bending_moment(x) for x in positions]

    response = {
        "Supports": supports,
        "reactions": reactions,
        "BeamDiagram": BeamDiagram,
        'deflection_data': [{'position': round(float(pos), 2), 'deflection': round(float(deflection), 6)}
                            for pos, deflection in zip(positions, deflections)],
        'shear_force_data': [{'position': round(float(pos), 2), 'shear_force': round(float(shear_force), 2)}
                             for pos, shear_force in zip(positions, shear_forces)],
        'bending_moment_data': [{'position': round(float(pos), 2), 'bending_moment': round(float(bending_moment), 2)}
                                for pos, bending_moment in zip(positions, bending_moments)]
    }

    return jsonify(response)

# ==================================================
# PART 2: PLAGIARISM CHECKER API
# ==================================================
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

CUSTOM_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

def check_plagiarism(text):
    if not API_KEY or not SEARCH_ENGINE_ID:
        return {"error": "Search API credentials (API_KEY/SEARCH_ENGINE_ID) are missing. Plagiarism check is disabled.", "total_percent": 0, "matches": []}

    sentences = re.split(r'(?<=[.!?]) +', text)
    total_chars = len(text)
    exact_chars = 0
    partial_chars = 0
    matches = []

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue

        try:
            resp = requests.get(
                CUSTOM_SEARCH_URL,
                params={"key": API_KEY, "cx": SEARCH_ENGINE_ID, "q": f'"{sentence}"', "num": 5},
                timeout=10
            )
            resp.raise_for_status()
            res = resp.json()

            if 'items' in res:
                best_match = None
                best_partial_score = 0

                for item in res['items']:
                    snippet = item.get('snippet', '').lower()
                    sentence_lower = sentence.lower()

                    # Exact match
                    if sentence_lower in snippet:
                        best_match = ("exact", item)
                        break

                    # Partial match using Jaccard Similarity
                    sentence_words = set(re.findall(r'\w+', sentence_lower))
                    snippet_words = set(re.findall(r'\w+', snippet))
                    similarity = jaccard_similarity(sentence_words, snippet_words)

                    if similarity >= 0.3 and similarity > best_partial_score:
                        best_match = ("partial", item)
                        best_partial_score = similarity

                if best_match:
                    match_type, item = best_match
                    char_count = len(sentence)

                    if match_type == "exact":
                        exact_chars += char_count
                    elif match_type == "partial":
                        partial_chars += char_count

                    matches.append({
                        "sentence": sentence,
                        "source": item['link'],
                        "snippet": item.get('snippet', '')[:200] + "...",
                        "match_type": match_type
                    })

        except requests.exceptions.RequestException as e:
            print(f"Search API Error: {e}")
        except Exception as e:
            print(f"Error: {e}")

    exact_percent = round((exact_chars / total_chars * 100)) if total_chars > 0 else 0
    partial_percent = round((partial_chars / total_chars * 100)) if total_chars > 0 else 0
    total_percent = round(exact_percent + partial_percent)

    return {
        "matches": matches,
        "exact_percent": exact_percent,
        "partial_percent": partial_percent,
        "total_percent": total_percent
    }

@app.route('/check_plagiarism', methods=['POST'])
def handle_check():
    try:
        user_ip = request.remote_addr
        today = datetime.today().date()

        # Reset counter daily
        if plagiarism_trials[user_ip]["date"] != today:
            plagiarism_trials[user_ip]["date"] = today
            plagiarism_trials[user_ip]["count"] = 0

        # Check usage limit
        if plagiarism_trials[user_ip]["count"] >= 100:
            return jsonify({
                "error": "You have reached the free limit of 100 plagiarism checks today. Please try again tomorrow."
            }), 429

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        if len(text) > 5000:
            return jsonify({"error": "Text exceeds 5000 characters"}), 400

        plagiarism_trials[user_ip]["count"] += 1
        result = check_plagiarism(text)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================================================
# PART 3: HUMANIZER API
# ==================================================
@app.route('/api/humanize', methods=['POST'])
def handle_humanize_text():
    allowed, err_response = check_ai_limit()
    if not allowed:
        return err_response, 429

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON body'}), 400

        text = data.get('text', '').strip()
        strength = data.get('strength', 'medium')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if len(text) > 10000:
            return jsonify({'error': 'Text too long (max 10,000 characters)'}), 400

        if strength not in ('light', 'medium', 'strong'):
            strength = 'medium'

        from humanizer import humanizer
        if not humanizer:
            return jsonify({'error': 'The Humanizer engine is currently warming up or encountered a dependency error. Please try again in 30 seconds.'}), 503

        analysis_before = humanizer.analyze_text(text)
        humanized = humanizer.humanize(text, strength=strength)
        analysis_after = humanizer.analyze_text(humanized)

        return jsonify({
            'success': True,
            'original': text,
            'humanized': humanized,
            'strength': strength,
            'analysis_before': analysis_before,
            'analysis_after': analysis_after,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================================================
# PART 4: AI DETECTOR API
# ==================================================
@app.route('/api/analyze', methods=['POST'])
def handle_ai_analyze():
    allowed, err_response = check_ai_limit()
    if not allowed:
        return err_response, 429

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
        
    text = data['text']
    result = analyze_text_ai(text)
    
    if not detector_classifier:
        return jsonify({"error": "AI Detector engine is still warming up (loading neural weights). Please try again in a few seconds."}), 503

    if not result:
        return jsonify({"error": "Analysis failed. Please check the input text and try again."}), 500
        
    return jsonify(result)

# ==================================================
# PART 5: GRAMMAR CHECKER API
# ==================================================
import urllib.request
import urllib.parse

LANGUAGETOOL_API_URL = "https://api.languagetool.org/v2/check"

@app.route('/api/grammar', methods=['POST'])
def handle_grammar_check():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    try:
        post_data = urllib.parse.urlencode({
            'text': text,
            'language': 'en-US'
        }).encode('utf-8')

        req = urllib.request.Request(LANGUAGETOOL_API_URL, data=post_data)
        req.add_header('User-Agent', 'OOKCalculatorGrammarChecker/1.0')
        req.add_header('Accept', 'application/json')

        with urllib.request.urlopen(req, timeout=15) as response:
            api_result = json.loads(response.read().decode('utf-8'))

    except Exception as e:
        print("Error calling LanguageTool:", e)
        return jsonify({'error': 'Failed to analyze text', 'details': str(e)}), 500

    errors = []
    if 'matches' in api_result:
        for match in api_result['matches']:
            errors.append({
                'message': match.get('message', ''),
                'replacements': [r['value'] for r in match.get('replacements', [])],
                'context': match.get('context', {}).get('text', ''),
                'offset': match.get('offset', 0),
                'errorLength': match.get('length', 0),
                'ruleId': match.get('rule', {}).get('id', ''),
                'category': match.get('rule', {}).get('category', {}).get('name', ''),
            })

    return jsonify({
        'original_text': text,
        'errors': errors
    })


# ==================================================
# PART 6: PARAPHRASING TOOL API (Google Gemini)
# ==================================================

@app.route('/api/paraphrase', methods=['POST'])
def handle_paraphrase():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data.get('text', '').strip()
    mode = data.get('mode', 'standard').lower()

    if not text:
        return jsonify({'error': 'Empty text provided'}), 400

    prompt = f"Paraphrase the following text. The goal is to make it sound '{mode}'. Return ONLY the newly paraphrased text without any extra conversation or markdown.\n\nText:\n{text}"
    
    result, error = call_gemini(prompt, system_instruction=f"You are a professional rewording assistant. Paraphrase text in a {mode} style. Do not output anything except the paraphrased text.")
    
    if error:
        return jsonify({'error': error}), 500

    return jsonify({'paraphrased': result.strip()})


# ==================================================
# PART 7: SUMMARIZER API (Google Gemini)
# ==================================================
@app.route('/api/summarize', methods=['POST'])
def handle_summarize():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data.get('text', '').strip()
    format_type = data.get('type', 'paragraph')  # 'paragraph' or 'bullets'
    length_pref = data.get('length', 'medium')   # 'short', 'medium', 'long'

    if not text:
        return jsonify({'error': 'Empty text provided'}), 400

    length_instruction = {
        'short': 'Provide a very concise summary (1-2 sentences).',
        'medium': 'Provide a standard length summary.',
        'long': 'Provide a detailed, comprehensive summary.'
    }.get(length_pref, 'Provide a standard summary.')

    format_instruction = "Return the summary as a cohesive paragraph."
    if format_type == 'bullets':
        format_instruction = "Return the summary as a list of bullet points starting with a standard bullet character (•)."

    prompt = f"Summarize the following text.\n\nInstructions:\n- {length_instruction}\n- {format_instruction}\n\nText to summarize:\n{text}"
    
    result, error = call_gemini(prompt, system_instruction="You are an expert summarization AI. Return ONLY the summarized content with no conversational filler.")
    
    if error:
        return jsonify({'error': error}), 500

    return jsonify({'summary': result.strip()})


# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
