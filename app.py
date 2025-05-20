# --------------------------
# COMBINED FLASK APP: Beam Deflection + Plagiarism Checker
# --------------------------

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import requests
from difflib import SequenceMatcher
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --------------------------
# Load Environment Variables (for Plagiarism)
# --------------------------
load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv('API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

# --------------------------
# Import Custom Beam Classes (Required for Beam Deflection)
# --------------------------
from indeterminatebeam import Beam, Support, PointLoadV, TrapezoidalLoad

# ==================================================
# PART 1: BEAM DEFLECTION API
# ==================================================
@app.route('/beam', methods=['POST'])
def handle_beam_deflection():
    # Extract input data
    length = request.json.get('length')
    point_loads = request.json.get('point_loads', [])
    distributed_loads = request.json.get('distributed_loads', [])
    supports = request.json.get('supports', [])
    youngmodules = request.json.get('youngmodules')
    area = request.json.get('area')
    inertia = request.json.get('inertia')

    # Initialize beam object
    beam = Beam(length, A=area, I=inertia, E=youngmodules)

    # Add supports
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

    # Add point loads
    for pl in point_loads:
        beam.add_loads(PointLoadV(pl['magnitude'] * -1, pl['position']))

    # Add distributed loads
    for dl in distributed_loads:
        beam.add_loads(TrapezoidalLoad(
            force=(dl['start_magnitude'] * -1, dl['end_magnitude'] * -1),
            span=(dl['start_position'], dl['end_position']),
            angle=90
        ))

    # Analyze the beam
    beam.analyse()

    # Get reactions at support positions
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

    # Prepare diagram data
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
    service = build("customsearch", "v1", developerKey=API_KEY)

    total_chars = len(text)
    exact_chars = 0
    partial_chars = 0
    matches = []

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue

        try:
            res = service.cse().list(
                q=f'"{sentence}"',
                cx=SEARCH_ENGINE_ID,
                num=5
            ).execute()

            if 'items' in res:
                best_match = None
                best_partial_score = 0

                for item in res['items']:
                    snippet = item.get('snippet', '').lower()
                    sentence_lower = sentence.lower()

                    # Exact match
                    if sentence_lower in snippet:
                        best_match = ("exact", item)
                        break  # No need to check further

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

        except HttpError as e:
            print(f"API Error: {e}")
        except Exception as e:
            print(f"Error: {e}")

    # Rounded to integers
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
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        if len(text) > 5000:
            return jsonify({"error": "Text exceeds 5000 characters"}), 400

        result = check_plagiarism(text)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
