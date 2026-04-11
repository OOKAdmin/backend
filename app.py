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
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

#--------------------------
# AI Detector and NLTK Initialization
#--------------------------
import threading

_nltk_ready = False  # Guard: only download NLTK data once per process
_nltk_downloading = False

def download_nltk_resources():
    global _nltk_ready, _nltk_downloading
    if _nltk_ready or _nltk_downloading:
        return
    _nltk_downloading = True
    
    import nltk
    import os
    
    # Bypass corrupted Render cache by forcing downloads into a local isolated directory
    custom_nltk_dir = os.path.join(os.getcwd(), 'nltk_data_isolated')
    os.makedirs(custom_nltk_dir, exist_ok=True)
    if custom_nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, custom_nltk_dir)
        
    resources = ['punkt', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except Exception:
            try:
                nltk.download(res, download_dir=custom_nltk_dir, quiet=True)
            except Exception:
                pass
    _nltk_ready = True
    _nltk_downloading = False

# Kick off NLTK download asynchronously on boot
threading.Thread(target=download_nltk_resources, daemon=True).start()

# ======================
# GEMINI CONFIG
# ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
else:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

def call_gemini(prompt):
    """Helper to call Google Gemini API."""
    if not GEMINI_API_KEY:
        return "Error: Gemini API key is missing."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"Error: {str(e)}"



def analyze_text_ai(text):
    """AI Detection using Gemini API for deep linguistic analysis."""
    if not text or not text.strip():
        return None
        
    prompt = f"""
    Analyze the following text for AI-generated patterns. Be precise.
    Text: {text}
    
    Return a valid JSON object ONLY with the following structure:
    {{
      "aiPercentage": (number 0-100),
      "mood": (string describing the tone),
      "segments": [
        {{ "text": "sentence 1", "type": "ai" or "human" or "ai-refined" or "human-ai" }},
        ...
      ]
    }}
    """
    response = call_gemini(prompt)
    try:
        # Extract JSON from markdown if necessary
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            clean_resp = ""
            parts = response.split("```")
            for i in range(1, len(parts), 2):
                clean_resp += parts[i]
            response = clean_resp.strip()
            
        return json.loads(response)
    except Exception as e:
        print(f"AI Detector JSON Parse Error: {e}")
        # Return a neutral result if parsing fails
        return {
            "aiPercentage": 0,
            "mood": "Detection Unreliable",
            "segments": [{"text": text, "type": "human"}]
        }






# Humanizer integration is now lazy-loaded inside the route.
humanizer = None

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

def check_plagiarism(text):
    """Plagiarism check using Google Custom Search API."""
    if not API_KEY or not SEARCH_ENGINE_ID:
        return {"error": "Search API credentials (API_KEY/SEARCH_ENGINE_ID) are missing.", "total_percent": 0, "matches": []}

    sentences = re.split(r'(?<=[.!?]) +', text)
    total_chars = len(text)
    exact_chars = 0
    partial_chars = 0
    matches = []
    
    CUSTOM_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue

        try:
            resp = requests.get(
                CUSTOM_SEARCH_URL,
                params={"key": API_KEY, "cx": SEARCH_ENGINE_ID, "q": f'"{sentence}"', "num": 3},
                timeout=10
            )
            res = resp.json()
            if 'items' in res:
                for item in res['items']:
                    snippet = item.get('snippet', '').lower()
                    if sentence.lower() in snippet:
                        exact_chars += len(sentence)
                        matches.append({"sentence": sentence, "source": item['link'], "match_type": "exact"})
                        break
                    elif jaccard_similarity(set(sentence.lower().split()), set(snippet.split())) > 0.6:
                        partial_chars += len(sentence)
                        matches.append({"sentence": sentence, "source": item['link'], "match_type": "partial"})
                        break
        except Exception as e:
            print(f"Plagiarism Search Error: {e}")

    exact_percent = int((exact_chars / total_chars) * 100) if total_chars > 0 else 0
    partial_percent = int((partial_chars / total_chars) * 100) if total_chars > 0 else 0
    
    return {
        "matches": matches,
        "exact_percent": exact_percent,
        "partial_percent": partial_percent,
        "total_percent": min(100, exact_percent + partial_percent)
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
        
    if getattr(threading.current_thread().name, "_nltk_ready", globals().get("_nltk_ready", False)) == False:
        if not _nltk_ready:
            return jsonify({"error": "AI Engine is warming up and caching offline dictionaries. Please try again in 30 seconds."}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
        
    text = data['text']
    result = analyze_text_ai(text)
    

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
# PART 6: PARAPHRASING TOOL API (Local - No API needed)
# ==================================================
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import wordnet, stopwords
    from nltk.tag import pos_tag
    _paraphrase_ready = True
except Exception:
    _paraphrase_ready = False

def _get_wordnet_pos(treebank_tag):
    """Map Penn Treebank POS tags to WordNet POS constants."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return None

def paraphrase_local(text, mode='standard'):
    """Paraphrasing using Gemini API."""
    prompt = f"Paraphrase the following text using a {mode} style. Maintain the original meaning but change the wording: {text}"
    return call_gemini(prompt)

@app.route('/api/paraphrase', methods=['POST'])
def handle_paraphrase():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data.get('text', '').strip()
    mode = data.get('mode', 'standard').lower()

    if not text:
        return jsonify({'error': 'Empty text provided'}), 400

    paraphrased = paraphrase_local(text, mode)
    return jsonify({'paraphrased': paraphrased})



# ==================================================
# PART 7: SUMMARIZER API (Local - sumy extractive)
# ==================================================
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    _sumy_ready = True
    print("[Success] Sumy (local summarizer) initialized!")
except Exception as e:
    _sumy_ready = False
    print(f"[Warning] Sumy not available: {e}")


def summarize_local(text, length_pref='medium', format_type='paragraph'):
    """Summarization using Gemini API."""
    prompt = f"Summarize the following text. Length: {length_pref}, Format: {format_type}. Text: {text}"
    return call_gemini(prompt)



@app.route('/api/summarize', methods=['POST'])
def handle_summarize():


    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data.get('text', '').strip()
    format_type = data.get('type', 'paragraph')
    length_pref = data.get('length', 'medium')

    if not text:
        return jsonify({'error': 'Empty text provided'}), 400
    
    if len(text.split()) < 20:
        return jsonify({'error': 'Please provide at least 20 words for a meaningful summary.'}), 400

    try:
        summary = summarize_local(text, length_pref, format_type)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': f'Summarization failed: {str(e)}'}), 500


# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
