import re
import random

# Initialize VADER (will be downloaded if missing in app.py)
# analyzer = SentimentIntensityAnalyzer() # Now lazy-loaded



class TextHumanizer:
    def __init__(self):
        self._analyzer = None
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return

        import nltk
        from nltk.corpus import wordnet
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk import pos_tag as _pos_tag
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        self.wordnet = wordnet
        self.sent_tokenize = sent_tokenize
        self.word_tokenize = word_tokenize
        self.pos_tag = _pos_tag
        self._analyzer = SentimentIntensityAnalyzer()
        
        # Setup the rest of the attributes

        # ── Formal → Casual phrase dictionary (multi-word first, then single) ──
        self.formal_to_casual = {
            'in order to': 'to',
            'due to the fact that': 'because',
            'at this point in time': 'now',
            'in the event that': 'if',
            'for the purpose of': 'to',
            'with respect to': 'about',
            'in accordance with': 'following',
            'in conjunction with': 'with',
            'as a result of': 'because of',
            'in spite of': 'despite',
            'take into consideration': 'consider',
            'make a decision': 'decide',
            'have the ability to': 'can',
            'be in a position to': 'can',
            'it is important to note': 'keep in mind',
            'it is worth noting': 'note that',
            'it should be noted': 'note that',
            'in conclusion': 'to wrap up',
            'to summarize': 'in short',
            'in today\'s world': 'nowadays',
            'in today\'s fast-paced world': 'nowadays',
            'prior to': 'before',
            'subsequent to': 'after',
            'in terms of': 'when it comes to',
            'with regard to': 'about',
            'utilize': 'use',
            'commence': 'start',
            'purchase': 'buy',
            'endeavor': 'try',
            'nevertheless': 'still',
            'therefore': 'so',
            'moreover': 'also',
            'furthermore': 'also',
            'subsequently': 'then',
            'approximately': 'about',
            'individual': 'person',
            'facilitate': 'help',
            'demonstrate': 'show',
            'establish': 'set up',
            'significant': 'important',
            'numerous': 'many',
            'currently': 'now',
            'regarding': 'about',
            'consequently': 'so',
            'obtain': 'get',
            'provide': 'give',
            'require': 'need',
            'ensure': 'make sure',
            'implement': 'carry out',
            'additional': 'more',
            'sufficient': 'enough',
            'initial': 'first',
            'attempt': 'try',
            'assist': 'help',
            'leverage': 'use',
            'optimal': 'best',
            'ascertain': 'find out',
            'formulate': 'create',
            'elucidate': 'explain',
            'ameliorate': 'improve',
            'terminate': 'end',
            'initiate': 'start',
            'comprehend': 'understand',
            'necessitate': 'require',
            'indicate': 'show',
            'pertaining': 'relating',
            'paramount': 'crucial',
            'pivotal': 'key',
            'robust': 'solid',
            'intricate': 'complex',
            'multifaceted': 'complex',
            'comprehensive': 'thorough',
            'innovative': 'new',
            'delve': 'dive',
            'underscore': 'highlight',
            'synergy': 'teamwork',
            'streamline': 'simplify',
            'holistic': 'overall',
            'transformative': 'life-changing',
            'revolutionary': 'major',
        }

        # ── AI "tell" phrases and replacements ──
        self.ai_tell_replacements = {
            'tapestry': 'mix',
            'realm': 'area',
            'nuanced approach': 'careful approach',
            'cutting-edge': 'latest',
            'game-changer': 'big deal',
            'scalable': 'flexible',
            'robust solution': 'solid solution',
            'seamlessly': 'smoothly',
            'unprecedented': 'rare',
            'state-of-the-art': 'modern',
        }

        # ── Contraction patterns ──
        self.contractions = {
            r'\bI am\b': "I'm",
            r'\byou are\b': "you're",
            r'\bhe is\b': "he's",
            r'\bshe is\b': "she's",
            r'\bit is\b': "it's",
            r'\bthey are\b': "they're",
            r'\bwe are\b': "we're",
            r'\bwill not\b': "won't",
            r'\bdo not\b': "don't",
            r'\bdoes not\b': "doesn't",
            r'\bdid not\b': "didn't",
            r'\bcould not\b': "couldn't",
            r'\bwould not\b': "wouldn't",
            r'\bshould not\b': "shouldn't",
            r'\bhave not\b': "haven't",
            r'\bhas not\b': "hasn't",
            r'\bhad not\b': "hadn't",
            r'\bI have\b': "I've",
            r'\byou have\b': "you've",
            r'\bthey have\b': "they've",
            r'\bwe have\b': "we've",
            r'\bI will\b': "I'll",
            r'\byou will\b': "you'll",
            r'\bhe will\b': "he'll",
            r'\bshe will\b': "she'll",
            r'\bthey will\b': "they'll",
            r'\bwe will\b': "we'll",
            r'\bI would\b': "I'd",
            r'\byou would\b': "you'd",
            r'\bthat is\b': "that's",
            r'\bthere is\b': "there's",
            r'\bwhat is\b': "what's",
            r'\bwho is\b': "who's",
            r'\bhere is\b': "here's",
            r'\bcannot\b': "can't",
            r'\bare not\b': "aren't",
            r'\bwas not\b': "wasn't",
            r'\bwere not\b': "weren't",
            r'\bis not\b': "isn't",
        }

        # ── Words that should never be replaced ──
        self.skip_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'and', 'but', 'or', 'nor',
            'so', 'for', 'yet', 'i', 'me', 'my', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'it', 'its', 'we', 'our', 'they', 'their',
            'them', 'what', 'which', 'who', 'this', 'that', 'these', 'those',
            'here', 'there', 'then', 'than', 'just', 'also', 'very', 'more',
            'most', 'some', 'any', 'all', 'not', 'no', 'up', 'out', 'if',
            'of', 'in', 'on', 'to', 'at', 'by', 'from', 'with', 'about',
            'into', 'through', 'over', 'now', 'only', 'when', 'where', 'why',
            'how', 'get', 'got', 'go', 'make', 'made', 'use', 'used', 'new',
            'old', 'big', 'good', 'bad', 'well', 'way', 'say', 'said', 'know',
        }

        # ── POS tag → WordNet POS mapping ──
        self.pos_map = {
            'JJ': wordnet.ADJ, 'JJR': wordnet.ADJ, 'JJS': wordnet.ADJ,
            'NN': wordnet.NOUN, 'NNS': wordnet.NOUN,
            'RB': wordnet.ADV, 'RBR': wordnet.ADV, 'RBS': wordnet.ADV,
        }

        # ── Casual sentence openers ──
        self.casual_openers = [
            "Honestly, ", "Look, ", "To be clear, ", "Here's the thing — ",
            "Basically, ", "Simply put, ", "The thing is, ", "To be fair, ",
            "In reality, ", "At the end of the day, ",
        ]

        # ── AI patterns for scoring ──
        self.ai_patterns = [
            'delve', 'tapestry', 'realm', 'it\'s important to note',
            'it\'s worth noting', 'in conclusion', 'to summarize',
            'comprehensive', 'intricate', 'multifaceted', 'nuanced',
            'in today\'s world', 'underscore', 'paramount', 'pivotal',
            'robust', 'leverage', 'synergy', 'innovative', 'cutting-edge',
            'game-changer', 'revolutionary', 'transformative', 'holistic',
            'streamline', 'scalable', 'furthermore', 'moreover', 'consequently',
            'therefore', 'nevertheless', 'in order to', 'it is important',
            'it should be noted', 'it is worth',
        ]

    # ─────────────────────────────────────────────
    #  WordNet synonym lookup
    # ─────────────────────────────────────────────
    def get_wordnet_synonyms(self, word, pos):
        """Return simpler synonyms from the full English WordNet dictionary."""
        self._initialize()
        synonyms = set()
        try:
            synsets = self.wordnet.synsets(word, pos=pos)
            for synset in synsets[:3]:
                for lemma in synset.lemmas():
                    name = lemma.name().replace('_', ' ')
                    if (name.lower() != word.lower()
                            and ' ' not in name
                            and name.isalpha()
                            and len(name) <= len(word) + 1):
                        synonyms.add(name)
        except Exception:
            pass
        # Sort by length ascending (shorter = more common/simpler)
        return sorted(synonyms, key=len)[:5]

    # ─────────────────────────────────────────────
    #  Pipeline steps
    # ─────────────────────────────────────────────
    def replace_ai_tells(self, text):
        for phrase, replacement in self.ai_tell_replacements.items():
            text = re.sub(re.escape(phrase), replacement, text, flags=re.IGNORECASE)
        return text

    def apply_formal_to_casual(self, text):
        # Multi-word phrases first (longer → shorter avoids partial matches)
        for formal, casual in sorted(self.formal_to_casual.items(), key=lambda x: -len(x[0])):
            if ' ' in formal:
                text = re.sub(re.escape(formal), casual, text, flags=re.IGNORECASE)
        for formal, casual in self.formal_to_casual.items():
            if ' ' not in formal:
                text = re.sub(rf'\b{re.escape(formal)}\b', casual, text, flags=re.IGNORECASE)
        return text

    def apply_contractions(self, text):
        for pattern, replacement in self.contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def wordnet_synonym_pass(self, text, strength=0.4):
        """Replace complex words using the full NLTK WordNet English dictionary."""
        self._initialize()
        try:
            words = self.word_tokenize(text)
            tagged = self.pos_tag(words)
            result = []
            for word, tag in tagged:
                wn_pos = self.pos_map.get(tag)
                if (not wn_pos
                        or len(word) < 6
                        or word.lower() in self.skip_words
                        or not word.isalpha()
                        or random.random() > strength):
                    result.append(word)
                    continue
                synonyms = self.get_wordnet_synonyms(word, wn_pos)
                if synonyms:
                    chosen = random.choice(synonyms[:3])
                    if word[0].isupper():
                        chosen = chosen.capitalize()
                    result.append(chosen)
                else:
                    result.append(word)
            return ' '.join(result)
        except Exception:
            return text

    def fix_tokenizer_spacing(self, text):
        """Repair spacing artifacts left by word_tokenize."""
        text = re.sub(r"\s+n't", "n't", text)
        text = re.sub(r"\s+'s\b", "'s", text)
        text = re.sub(r"\s+'re\b", "'re", text)
        text = re.sub(r"\s+'ve\b", "'ve", text)
        text = re.sub(r"\s+'ll\b", "'ll", text)
        text = re.sub(r"\s+'d\b", "'d", text)
        text = re.sub(r"\s+'m\b", "'m", text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def vary_sentence_structure(self, text, strength='medium'):
        """Break overly long sentences and add natural variety."""
        self._initialize()
        try:
            sentences = self.sent_tokenize(text)
        except Exception:
            return text

        if len(sentences) < 2:
            return text

        processed = []
        for sent in sentences:
            words = sent.split()
            # Break very long sentences at a comma near the middle
            if len(words) > 22 and strength in ('medium', 'strong'):
                mid_start = len(sent) // 3
                mid_end = 2 * len(sent) // 3
                comma_pos = sent.find(', ', mid_start, mid_end)
                if comma_pos > 0:
                    part1 = sent[:comma_pos].strip() + '.'
                    part2 = sent[comma_pos + 2:].strip()
                    if part2:
                        part2 = part2[0].upper() + part2[1:]
                    processed.extend([part1, part2])
                    continue
            processed.append(sent)

        return ' '.join(processed)

    def add_natural_touches(self, text, strength='medium'):
        """Sprinkle casual sentence openers to break AI monotony."""
        self._initialize()
        try:
            sentences = self.sent_tokenize(text)
        except Exception:
            return text

        if len(sentences) < 2:
            return text

        opener_chance = {'light': 0.12, 'medium': 0.22, 'strong': 0.35}.get(strength, 0.22)
        result = [sentences[0]]  # never touch the first sentence

        for sent in sentences[1:]:
            if (random.random() < opener_chance
                    and len(sent.split()) > 6
                    and not any(sent.startswith(op.strip()) for op in self.casual_openers)):
                opener = random.choice(self.casual_openers)
                sent = opener + sent[0].lower() + sent[1:]
            result.append(sent)

        return ' '.join(result)

    # ─────────────────────────────────────────────
    #  Analysis helpers
    # ─────────────────────────────────────────────
    def _syllables(self, word):
        word = word.lower()
        count, prev_vowel = 0, False
        for ch in word:
            v = ch in 'aeiouy'
            if v and not prev_vowel:
                count += 1
            prev_vowel = v
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)

    def calculate_readability(self, text):
        self._initialize()
        try:
            sents = self.sent_tokenize(text)
            words = [w for w in self.word_tokenize(text) if w.isalpha()]
            if not sents or not words:
                return 50
            asl = len(words) / len(sents)
            asw = sum(self._syllables(w) for w in words) / len(words)
            score = 206.835 - 1.015 * asl - 84.6 * asw
            return min(100, max(0, round(score)))
        except Exception:
            return 50

    def estimate_ai_score(self, text):
        self._initialize()
        score = 20
        tl = text.lower()
        for pat in self.ai_patterns:
            if pat in tl:
                score += 7
        try:
            sents = self.sent_tokenize(text)
            if len(sents) >= 3:
                lengths = [len(s.split()) for s in sents]
                avg = sum(lengths) / len(lengths)
                variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
                if variance < 10:
                    score += 15
        except Exception:
            pass
        contractions_count = len(re.findall(r"\b\w+'\w+\b", text))
        if len(text.split()) > 20 and contractions_count == 0:
            score += 12
        return min(100, max(0, score))

    def estimate_formality(self, text):
        words = text.lower().split()
        formal_hits = sum(1 for w in words if w in self.formal_to_casual)
        base = min(80, int((formal_hits / max(len(words), 1)) * 600))
        if re.search(r'\b(therefore|consequently|furthermore|moreover|nevertheless)\b', text, re.I):
            base = min(100, base + 20)
        return base

    def analyze_text(self, text):
        self._initialize()
        try:
            sents = self.sent_tokenize(text)
            words = text.split()
            # Use VADER instead of TextBlob for sentiment
            sentiment_scores = self._analyzer.polarity_scores(text)
            
            return {
                'words': len(words),
                'sentences': len(sents),
                'avg_sentence_length': round(len(words) / max(len(sents), 1), 1),
                'sentiment': round(sentiment_scores['compound'], 2),
                'readability': self.calculate_readability(text),
                'ai_score': self.estimate_ai_score(text),
                'formality': self.estimate_formality(text),
            }
        except Exception:
            return {
                'words': len(text.split()), 'sentences': 1,
                'avg_sentence_length': len(text.split()),
                'sentiment': 0, 'readability': 50, 'ai_score': 50, 'formality': 50,
            }

    # ─────────────────────────────────────────────
    #  Main humanize pipeline
    # ─────────────────────────────────────────────
    def humanize(self, text, strength='medium'):
        if len(text.strip()) < 3:
            return text

        wn_chance = {'light': 0.25, 'medium': 0.45, 'strong': 0.65}.get(strength, 0.45)

        # 1. Remove obvious AI tells
        text = self.replace_ai_tells(text)

        # 2. Formal phrases → casual equivalents
        text = self.apply_formal_to_casual(text)

        # 3. WordNet full-dictionary synonym pass
        text = self.wordnet_synonym_pass(text, strength=wn_chance)
        text = self.fix_tokenizer_spacing(text)

        # 4. Contractions (very impactful for human feel)
        text = self.apply_contractions(text)

        # 5. Sentence structure variation
        text = self.vary_sentence_structure(text, strength)

        # 6. Natural human touches
        text = self.add_natural_touches(text, strength)

        # 7. Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r' ([.,!?;:])', r'\1', text)
        text = re.sub(r'\.{3,}', '...', text)

        return text


humanizer = TextHumanizer()