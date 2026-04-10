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
        
    resources = [
        'punkt', 'punkt_tab', 'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4', 'vader_lexicon', 'stopwords'
    ]
    for res in resources:
        try:
            # Check custom directory
            nltk.data.find(f"tokenizers/{res}") if 'punkt' in res else nltk.data.find(f"corpora/{res}")
        except Exception:
            try:
                nltk.download(res, download_dir=custom_nltk_dir, quiet=True)
            except Exception:
                pass
    _nltk_ready = True
    _nltk_downloading = False

# Kick off NLTK download asynchronously on boot
threading.Thread(target=download_nltk_resources, daemon=True).start()


print("Initializing AI components via Google Gemini API...")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# call_gemini removed - replaced with local, API-free implementations


def analyze_text_ai(text):
    """
    Local, API-free AI detection using heuristic linguistic analysis.
    Scores text on several features commonly associated with AI-generated content:
    - Low lexical diversity (repetitive vocabulary)
    - Low punctuation variation
    - High average sentence length
    - Low usage of first-person pronouns and contractions
    - Uniformly smooth sentence length variation
    """
    if not text or not text.strip():
        return None

    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text.lower())
    alpha_words = [w for w in words if w.isalpha()]
    
    if not alpha_words:
        return None

    # Feature 1: Type-Token Ratio (low TTR = repetitive = AI-like)
    ttr = len(set(alpha_words)) / len(alpha_words)  # ideal human ~0.7+
    ttr_score = max(0, min(100, int((1.0 - ttr) * 100)))

    # Feature 2: Average sentence length (very uniform/long = AI-like)
    sent_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
    avg_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
    # AI sentences tend to be 18-30 words. Above 25 is suspicious.
    len_score = max(0, min(100, int((avg_len - 10) * 4))) if avg_len > 15 else 0

    # Feature 3: Sentence length variance (AI is very uniform, humans vary)
    if len(sent_lengths) > 1:
        mean_l = avg_len
        variance = sum((l - mean_l) ** 2 for l in sent_lengths) / len(sent_lengths)
        # Low variance = uniform = AI-like
        uniformity_score = max(0, min(100, int(100 - variance)))
    else:
        uniformity_score = 70  # single sentence, assume suspicious

    # Feature 4: First-person & contraction usage (humans use these more)
    personal_pronouns = ['i', 'me', 'my', 'mine', 'we', 'our', 'us']
    contractions = ["n't", "'re", "'ve", "'ll", "'d", "'m"]
    personal_count = sum(1 for w in words if w in personal_pronouns)
    contraction_count = sum(1 for w in words if w in contractions)
    human_signals = (personal_count + contraction_count) / max(1, len(alpha_words)) * 100
    human_score = max(0, min(100, int(human_signals * 20)))  # amplify signal

    # Weighted average AI probability
    ai_probability = (
        ttr_score * 0.35 +
        len_score * 0.20 +
        uniformity_score * 0.30 +
        (100 - human_score) * 0.15
    )
    final_ai_score = max(0, min(100, int(ai_probability)))

    # Mood classification
    if final_ai_score >= 80:
        mood = "AI Based"
    elif final_ai_score >= 60:
        mood = "AI Based & AI Refined"
    elif final_ai_score >= 40:
        mood = "Human Written & AI Refined"
    else:
        mood = "Human Written"

    # Sentence-level classification
    segments = []
    for s in sentences:
        s_words = nltk.word_tokenize(s.lower())
        s_alpha = [w for w in s_words if w.isalpha()]
        if not s_alpha:
            continue
        s_ttr = len(set(s_alpha)) / len(s_alpha)
        s_personal = sum(1 for w in s_words if w in personal_pronouns)
        s_contractions = sum(1 for w in s_words if w in contractions)
        s_human_signals = (s_personal + s_contractions) / max(1, len(s_alpha)) * 100
        s_ai_score = max(0, min(100, int((1.0 - s_ttr) * 70 + (100 - s_human_signals * 20) * 0.30)))

        if s_ai_score >= 75:
            sType = "ai"
        elif s_ai_score >= 55:
            sType = "ai-refined"
        elif s_ai_score >= 35:
            sType = "human-ai"
        else:
            sType = "human"

        segments.append({"text": s.strip(), "type": sType})

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

def check_plagiarism(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    total_chars = len(text)
    exact_chars = 0
    partial_chars = 0
    matches = []
    
    if total_chars < 20:
        return {"error": "Text is too short.", "total_percent": 0, "matches": []}

    try:
        from googlesearch import search
    except ImportError:
        return {"error": "Local search module is not installed.", "total_percent": 0, "matches": []}

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue

        try:
            results = list(search(f'"{sentence}"', num_results=3, sleep_interval=1, advanced=True))
            
            match_found = False
            for item in results:
                snippet = getattr(item, 'description', '')
                url = getattr(item, 'url', '')
                if not snippet or not url:
                    continue
                    
                if sentence.lower() in snippet.lower():
                    exact_chars += len(sentence)
                    matches.append({"sentence": sentence, "source": url, "snippet": snippet[:200] + "...", "match_type": "exact"})
                    match_found = True
                    break
                elif jaccard_similarity(set(sentence.lower().split()), set(snippet.lower().split())) > 0.6:
                    partial_chars += len(sentence)
                    matches.append({"sentence": sentence, "source": url, "snippet": snippet[:200] + "...", "match_type": "partial"})
                    match_found = True
                    break
                    
            if not match_found:
                results_broad = list(search(sentence, num_results=2, sleep_interval=1, advanced=True))
                for item in results_broad:
                    snippet = getattr(item, 'description', '')
                    url = getattr(item, 'url', '')
                    if snippet and url and jaccard_similarity(set(sentence.lower().split()), set(snippet.lower().split())) > 0.6:
                        partial_chars += len(sentence)
                        matches.append({"sentence": sentence, "source": url, "snippet": snippet[:200] + "...", "match_type": "partial"})
                        break

        except Exception as e:
            print(f"Search API error on sentence: {e}")
            continue

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
    """Local paraphrasing using NLTK WordNet synonym substitution."""
    try:
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import wordnet, stopwords
        from nltk.tag import pos_tag
    except:
        return text

    stop_words = set(stopwords.words('english'))
    
    # Mode-based substitution rates
    mode_rates = {
        'standard': 0.30,
        'fluency': 0.25,
        'creative': 0.50,
        'formal': 0.35
    }
    sub_rate = mode_rates.get(mode, 0.30)
    
    sentences = sent_tokenize(text)
    result_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged = pos_tag(words)
        new_words = []
        sub_count = 0
        max_subs = max(1, int(len(words) * sub_rate))

        for word, tag in tagged:
            wn_pos = _get_wordnet_pos(tag)
            # Only replace non-stopword content words with a synonym
            if sub_count < max_subs and wn_pos and word.lower() not in stop_words and word.isalpha() and len(word) > 3:
                synsets = wordnet.synsets(word, pos=wn_pos)
                candidates = []
                for syn in synsets:
                    for lemma in syn.lemmas():
                        candidate = lemma.name().replace('_', ' ')
                        if candidate.lower() != word.lower() and candidate.isalpha():
                            candidates.append(candidate)
                if candidates:
                    best = candidates[0]
                    # Preserve original capitalization
                    if word[0].isupper():
                        best = best.capitalize()
                    new_words.append(best)
                    sub_count += 1
                    continue
            new_words.append(word)

        # Re-join tokens (simple)
        result = ' '.join(new_words)
        # Fix spacing before punctuation
        result = re.sub(r'\s+([.,!?;:\'"\)])', r'\1', result)
        result = re.sub(r'([\(])\s+', r'\1', result)
        result_sentences.append(result)

    return ' '.join(result_sentences)


@app.route('/api/paraphrase', methods=['POST'])
def handle_paraphrase():
    if not _nltk_ready:
        return jsonify({"error": "Paraphrase tool is warming up and caching offline dictionaries. Please try again in 30 seconds."}), 503

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
    """Local extractive summarization using sumy LSA."""
    import math
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lsa import LsaSummarizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words
    except:
        # Fallback: return first 2 sentences
        import nltk
        sents = nltk.sent_tokenize(text)
        return ' '.join(sents[:2])

    LANGUAGE = "english"
    length_map = {'short': 2, 'medium': 4, 'long': 7}
    sentence_count = length_map.get(length_pref, 4)
    
    # Guard: don't request more sentences than exist
    import nltk
    all_sents = nltk.sent_tokenize(text)
    sentence_count = min(sentence_count, max(1, len(all_sents)))

    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    summary_sentences = summarizer(parser.document, sentence_count)
    result = ' '.join(str(s) for s in summary_sentences)

    if format_type == 'bullets':
        bullet_sents = [str(s).strip() for s in summary_sentences if str(s).strip()]
        result = '\n'.join(f'\u2022 {s}{"" if s.endswith(".") else "."}' for s in bullet_sents)

    return result


@app.route('/api/summarize', methods=['POST'])
def handle_summarize():
    if not _nltk_ready:
        return jsonify({"error": "Summarizer tool is warming up and caching offline dictionaries. Please try again in 30 seconds."}), 503

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
