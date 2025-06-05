from flask import Flask, request, jsonify, render_template
from word2number import w2n

from flask import Flask, request, jsonify
import whisper
import os
import re
import yt_dlp
from urllib.parse import urlparse, parse_qs
from transformers import pipeline
import webvtt  # NEW: for parsing subtitles
from faster_whisper import WhisperModel
import torch


app = Flask(__name__)
import sqlite3

# Connect to (or create) SQLite database file
conn = sqlite3.connect('feedback.db', check_same_thread=False)
cursor = conn.cursor()

# Create feedback table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    video_url TEXT,
    summary TEXT,
    feedback INTEGER, -- 1 for positive, 0 for negative
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()

# Initialize models
# Load model with GPU support
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"🚀 Loading Faster-Whisper on {device.upper()}...")
    return WhisperModel("base", device=device, compute_type=compute_type)

whisper_model = load_whisper_model()

from transformers import pipeline

# Standard summarizer for default summaries
summarizer_standard = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Creative summarizer that supports prompts
summarizer_prompt = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")




DOWNLOAD_FOLDER = "downloads"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# More precise regex to catch math expressions involving digits, letters, and math operators
FORMULA_REGEX = re.compile(r"([A-Za-z0-9]+(?:[\+\-\*/\^][A-Za-z0-9]+)+)")

# ----------- Helper Functions -----------
def save_feedback(user_id, video_url, summary, feedback):
    cursor.execute('''
        INSERT INTO user_feedback (user_id, video_url, summary, feedback)
        VALUES (?, ?, ?, ?)
    ''', (user_id, video_url, summary, feedback))
    conn.commit()


def clean_youtube_url(raw_url):
    parsed = urlparse(raw_url)
    if "youtu.be" in parsed.netloc:
        video_id = parsed.path.strip("/")
    elif "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        video_id = query.get("v", [None])[0]
    else:
        raise ValueError("Unsupported YouTube URL format")
    if not video_id:
        raise ValueError("Could not extract video ID from URL")
    return f"https://www.youtube.com/watch?v={video_id}"

def download_youtube_audio(url):
    print(f"🎯 Downloading from URL: {url}")
    output_path = os.path.join(DOWNLOAD_FOLDER, "audio.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',

        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
        'noplaylist': True,
        'ffmpeg_location': r'C:\Users\SAMIKSHA\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin',  # <-- update this to your actual ffmpeg bin path
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise RuntimeError(f"yt-dlp download failed: {e}")
    return os.path.join(DOWNLOAD_FOLDER, "audio.wav")

def transcribe_audio(audio_path):
    print("🔍 Transcribing with Faster-Whisper...")
    segments, info = whisper_model.transcribe(audio_path)
    return " ".join([segment.text for segment in segments])



# NEW: Download English subtitles (auto/manual) as VTT file
def download_captions(url):
    subtitle_path = os.path.join(DOWNLOAD_FOLDER, "subtitle.vtt")
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'outtmpl': os.path.join(DOWNLOAD_FOLDER, 'subtitle.%(ext)s'),
        'quiet': True,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subs = info.get('subtitles') or info.get('automatic_captions')
        if subs and 'en' in subs:
            ydl.download([url])
            if os.path.exists(subtitle_path):
                return subtitle_path
    return None


def has_subtitles(url):
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        subtitles = info.get('subtitles') or {}
        automatic_captions = info.get('automatic_captions') or {}
        return ('en' in subtitles) or ('en' in automatic_captions)


def download_youtube_subtitles(url):
    print(f"🎯 Downloading subtitles for URL: {url}")
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitlesformat': 'vtt',
        'outtmpl': os.path.join(DOWNLOAD_FOLDER, 'subs.%(ext)s'),
        'quiet': True,
        'noplaylist': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Try to find the subtitle file downloaded (could be subs.en.vtt or subs.vtt)
    for fname in os.listdir(DOWNLOAD_FOLDER):
        if fname.startswith("subs") and fname.endswith(".vtt"):
            subs_path = os.path.join(DOWNLOAD_FOLDER, fname)
            print(f"Found subtitles file: {subs_path}")
            return subs_path

    raise RuntimeError("Subtitles file not found.")

def words_to_math_symbols(expr):
    expr = expr.lower().strip()

    # Replace powers
    expr = re.sub(r'(\w+)\s+squared', r'\1^2', expr)
    expr = re.sub(r'(\w+)\s+cubed', r'\1^3', expr)

    # Replace root
    expr = re.sub(r'root\s+(\w+)', r'\\sqrt{\1}', expr)

    # Basic operators
    expr = re.sub(r'\bplus\b', '+', expr)
    expr = re.sub(r'\bminus\b', '-', expr)
    expr = re.sub(r'\btimes\b|\bmultiplied by\b', r'\\times', expr)
    expr = re.sub(r'\bdivided by\b', r'\\div', expr)

    # Fractions with over
    if " over " in expr:
        parts = expr.split(" over ")
        if len(parts) == 2:
            numerator = parts[0].strip()
            denominator = parts[1].strip()
            return f"\\frac{{{numerator}}}{{{denominator}}}"

    return expr



# NEW: Parse VTT subtitle file into plain text
def read_subtitle_text(vtt_path):
    lines = []
    for caption in webvtt.read(vtt_path):
        lines.append(caption.text)
    return "\n".join(lines)


def extract_formulas(text):
    formulas = set()
    text = text.lower()
    text = convert_word_math_to_symbols(text)
    # --- Canonical pi extraction ---
    pi_def_pattern = r'pi\s*(is|equals|=|is approximately|is about|≈|~)?\s*(approximately|equal to|equal|about|)?\s*(=|≈|~|is|equals|is approximately|is about|)?\s*(22\s*/\s*7|3\.14|3,14|3\s*\.\s*14|22\s*over\s*7)(?:\s*(or|,|and)\s*(3\.14|3,14|3\s*\.\s*14|22\s*/\s*7|22\s*over\s*7))?'
    pi_matches = re.findall(pi_def_pattern, text)
    if pi_matches:
        # Only add if the value is present in the match
        for match in pi_matches:
            value = match[3].replace(' ', '')
            if value in ['22/7', '3.14', '3,14', '3.14', '22over7']:
                if '22/7' in value or '22over7' in value:
                    formulas.add('pi ≈ 22/7')
                if '3.14' in value or '3,14' in value:
                    formulas.add('pi ≈ 3.14')
    else:
        pi_patterns = [
            r'pi\s*(is|equals|=|is approximately|is about|≈|~)\s*(approximately\s*)?(22\s*/\s*7|3\.14|3,14|3\s*\.\s*14|22\s*over\s*7)',
            r'pi\s*(≈|~|=|equals|is)\s*(22\s*/\s*7|3\.14|3,14|3\s*\.\s*14|22\s*over\s*7)',
            r'pi\s*(is|equals|=|is approximately|is about|≈|~)\s*(approximately\s*)?(3\.14|3,14|3\s*\.\s*14)',
        ]
        for pat in pi_patterns:
            m = re.search(pat, text)
            if m:
                value = m.group(m.lastindex).replace(' ', '') if m.lastindex else ''
                if value in ['22/7', '3.14', '3,14', '3.14', '22over7']:
                    if '22/7' in value or '22over7' in value:
                        formulas.add('pi ≈ 22/7')
                    if '3.14' in value or '3,14' in value:
                        formulas.add('pi ≈ 3.14')
                break
    # --- NEW: Extract formulas after 'the formula of ... is' or 'formula is' ---
    formula_callout_patterns = [
        r'the formula of [^\n\r:]* is ([^\n\r\.;,]+)',
        r'formula is ([^\n\r\.;,]+)'
    ]
    for pat in formula_callout_patterns:
        for match in re.finditer(pat, text):
            candidate = match.group(1).strip()
            # Only add if it looks like a formula (contains = and at least one variable)
            if '=' in candidate and re.search(r'[a-z]', candidate):
                # Remove trailing explanations
                cleaned = candidate.split('.')[0].split(',')[0].split('(')[0].strip()
                formulas.add(cleaned)
    # Only match formulas with at least one variable (not just numbers)
    formula_pattern = r'\b([a-z][a-z0-9_\^]*)\s*=\s*([a-z0-9_\^\s\+\-\*/^=×÷\.]+)\b'
    for match in re.finditer(formula_pattern, text):
        formula = match.group(0)
        left, right = re.split(r'\s*=\s*', formula, maxsplit=1)
        # Remove trailing explanations (after '.', ',', '(', 'since', 'so', 'because', etc.)
        cleaned = formula.split('.')[0].split(',')[0].split('(')[0].split('  ')[0].strip()
        cleaned = re.sub(r'\b(since|so|because|as|where|when|if|let|then|which|that|with|for|by|to|from|after|before|while|although|though|even|however|but|and|or|also|such|other|some|any|all|each|every|no|yes|one|two|three|four|five|six|seven|eight|nine|zero)\b.*$', '', cleaned).strip()
        if '=' in cleaned:
            left_clean, right_clean = [s.strip() for s in cleaned.split('=', 1)]
            # Exclude if right side is just a vague phrase (not a number/variable)
            vague_rights = {'approximately equal', 'approximately', 'equal', 'about', 'equal to', 'is', 'equals', 'is approximately', 'is about', '≈', '~'}
            if right_clean in vague_rights:
                continue
            # Exclude if either side is not a valid variable/expression (e.g., single word, 'example', etc.)
            if (re.fullmatch(r'[a-z][a-z0-9_\^]*', left_clean) or re.search(r'[a-z]', right_clean)):
                if not (left_clean.replace('.', '', 1).isdigit() and right_clean.replace('.', '', 1).isdigit()):
                    if not re.match(r'^\d+(\.\d+)?\s*[a-z]+', right_clean):
                        if left_clean not in {'example', 'th', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'being', 'been'} and right_clean not in {'example', 'th', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'being', 'been'}:
                            # Exclude verbose pi explanations (already handled above)
                            if not re.match(r'pi\s*=.*(22/7|3.14)', cleaned):
                                if len(left_clean) > 1 and len(right_clean) > 1:
                                    formulas.add(cleaned)
    # Add advanced math patterns (integrals, limits, etc.)
    advanced_patterns = [
        r'integral from [^ ]+ to [^ ]+ of [^ ]+ d[a-z]',
        r'limit as [a-z] approaches [^ ]+ of [^\.\n]+',
        r'square root of [^\s]+',
        r'cube root of [^\s]+',
        r'log of [^\s]+',
        r'ln of [^\s]+',
        r'sum from [^ ]+ to [^ ]+ of [^\.\n ]+',
        r'product from [^ ]+ to [^ ]+ of [^\.\n ]+',
    ]
    for pat in advanced_patterns:
        for match in re.findall(pat, text):
            formulas.add(match.strip())
    # Return sorted, unique formulas
    return sorted(set(f for f in formulas if len(f) > 2), key=len, reverse=True)

def convert_word_math_to_symbols(text):
    text = text.lower()
    # Convert number words to digits using word2number (w2n)
    def replace_number_words(match):
        try:
            return str(w2n.word_to_num(match.group(0)))
        except Exception:
            return match.group(0)
    # This regex matches number words and hyphenated/compound numbers
    text = re.sub(r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                  r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                  r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                  r'eighty|ninety|hundred|thousand)(?:[\s-](zero|one|two|three|four|five|six|seven|eight|nine))?\b',
                  replace_number_words, text)
    # Replace math operators and keywords
    replacements = [
        (r'is equal to|equals|is', '='),
        (r'plus', '+'),
        (r'minus', '-'),
        (r'times|multiplied by', '*'),
        (r'divided by|over', '/'),
        (r'squared', '^2'),
        (r'cubed', '^3'),
        (r'to the power of (\d+)', r'^\1'),
        (r'square root of ([a-z0-9]+)', r'sqrt(\1)'),
        (r'cube root of ([a-z0-9]+)', r'cbrt(\1)'),
        (r'integral from ([^ ]+) to ([^ ]+) of ([^ ]+) d([a-z])', r'int_{\1}^{\2} \3 d\4'),
        (r'limit as ([a-z]) approaches ([^ ]+) of ([^\.\n]+)', r'lim_{\1 \to \2} \3'),
    ]
    for pat, repl in replacements:
        text = re.sub(pat, repl, text)
    return text

def clean_text(text):
    """Normalize spaces, lowercase, remove punctuation except math symbols."""
    text = re.sub(r"\s+", " ", text)              # Collapse multiple spaces
    text = re.sub(r"[^\w\s=+\-*/^0-9.]", "", text) # Remove most punctuation
    return text.lower().strip()

def extract_examples(text):
    """
    Extracts step-by-step numeric examples from text, grouping consecutive lines with numeric calculations and short explanations.
    Filters out lines that contain 'th=' as a whole word, but does NOT filter out 'the'.
    Returns a list of examples, each possibly multi-line.
    """
    examples = []
    text = text.lower()
    text = convert_word_math_to_symbols(text)
    lines = re.split(r'[\n\r]+', text)
    current_example = []
    blank_count = 0
    example_pattern = r'\b(\d+(?:\s*[=\+\-\*/^]\s*\d+)+)\b'
    allowed_expl = set(['so', 'then', 'thus', 'therefore', 'step', 'answer', 'solution', 'hence', 'now', 'let', 'if', 'since', 'because', 'as', 'to', 'for', 'by', 'from', 'after', 'before', 'while', 'although', 'though', 'even', 'however', 'but', 'and', 'or', 'also'])
    def is_th_equals(line):
        # Only match 'th=' as a whole word, not as a substring (so 'the' is not matched)
        return bool(re.search(r'\bth=\b', line))
    for line in lines:
        line = line.strip()
        if not line:
            blank_count += 1
            if blank_count >= 2 and current_example:
                filtered = [l for l in current_example if not is_th_equals(l)]
                if filtered:
                    examples.append(' '.join(filtered))
                current_example = []
            continue
        blank_count = 0
        found = False
        for match in re.finditer(example_pattern, line):
            expr = match.group(0)
            if all(part.strip().replace('.', '', 1).isdigit() for part in re.split(r'[=\+\-\*/^]', expr) if part.strip()):
                found = True
        # Allow short explanation lines within an example
        if found or (len(line.split()) <= 5 and any(word in line for word in allowed_expl)) and not is_th_equals(line):
            current_example.append(line)
        else:
            if current_example:
                filtered = [l for l in current_example if not is_th_equals(l)]
                if filtered:
                    examples.append(' '.join(filtered))
                current_example = []
    if current_example:
        filtered = [l for l in current_example if not is_th_equals(l)]
        if filtered:
            examples.append(' '.join(filtered))
    # Remove duplicates and sort by length (longest first)
    examples = sorted(set(e for e in examples if len(e) > 2), key=len, reverse=True)
    return examples
from flask import request, jsonify

# --- RL summary stats: match keys to summary types ---
summary_stats = {
    "short": {"count": 0, "total_reward": 0},
    "medium": {"count": 0, "total_reward": 0},
    "long": {"count": 0, "total_reward": 0},
    "formal": {"count": 0, "total_reward": 0},
    "simple": {"count": 0, "total_reward": 0},
}

def select_summary_variant(epsilon=0.1):
    import random
    if random.random() < epsilon:
        # Explore
        return random.choice(list(summary_stats.keys()))
    else:
        # Exploit
        avg_rewards = {k: (v["total_reward"] / v["count"] if v["count"] > 0 else 0) for k,v in summary_stats.items()}
        return max(avg_rewards, key=avg_rewards.get)

def update_summary_stats(variant, reward):
    if variant in summary_stats:
        stat = summary_stats[variant]
        stat["count"] += 1
        stat["total_reward"] += reward

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()

    user_id = data.get('user_id', 'anonymous')
    video_url = data.get('video_url', '')
    summary = data.get('summary', '')  # Use .get() to avoid KeyError
    user_feedback = data.get('feedback')
    summary_type = data.get('summary_type', None)  # Optional
    variant_used = data.get('variant')  # NEW: Used for RL update

    # Log for debugging
    print(f"User: {user_id}, Video: {video_url}, Feedback: {user_feedback}, Summary: {summary[:30]}...")

    try:
        # Save feedback to DB
        save_feedback(user_id, video_url, summary, user_feedback)

        # Update RL stats using either summary_type or variant
        if summary_type and user_feedback is not None:
            update_summary_stats(summary_type, int(user_feedback))
        elif variant_used and user_feedback is not None:
            update_summary_stats(variant_used, int(user_feedback))

    except Exception as e:
        print(f"Error saving feedback to DB: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "success"})


# --- Main summary endpoint: use RL to select personalized summary ---
@app.route("/youtube", methods=["POST"])
def youtube_summary():
    try:
        data = request.get_json()
        raw_url = data.get("url")
        summary_type_raw = data.get("summary_type", "summary_medium")
        summary_type = summary_type_raw.replace("summary_", "")


        if not raw_url:
            return jsonify({"error": "No URL provided"}), 400

        url = clean_youtube_url(raw_url)

        # Prefer subtitles over transcription if available
        if not has_subtitles(url):
            print("✅ Using subtitles instead of transcription.")
            subs_path = download_youtube_subtitles(url)
            combined_text = read_subtitle_text(subs_path)
        else:
            print("🎧 Subtitles not available — falling back to audio transcription.")
            audio_path = download_youtube_audio(url)
            transcript = transcribe_audio(audio_path)
            combined_text = transcript

        # Extract formulas and examples
        print("--- Combined Text for Extraction (first 500 chars) ---\n", combined_text[:500])
        formulas = extract_formulas(combined_text)
        print("Extracted formulas:", formulas)
        examples = extract_examples(combined_text)
        print("Extracted examples:", examples)

        # Generate only the requested summary type
        summary_generators = {
            "short": summarize_text_short,
            "medium": summarize_text_medium,
            "long": summarize_text_long,
            "formal": summarize_text_formal,
            "simple": summarize_text_simplified
        }

        if summary_type not in summary_generators:
            return jsonify({"error": f"Invalid summary type: {summary_type}"}), 400

        summary = summary_generators[summary_type](combined_text)

        return jsonify({
            "summary": summary,
            "summary_type": summary_type,
            "formulas": [f"\\({words_to_math_symbols(f)}\\)" for f in formulas],
            "examples": examples
        })

    except Exception as e:
        print("🔥 Error:", str(e))
        return jsonify({"error": str(e)}), 500


# ----------- Routes -----------

def safe_summarize(text, max_len, min_len):
    if not text.strip():
        print("safe_summarize: Empty input text.")
        return ""
    try:
        result = summarizer_standard(text, max_length=max_len, min_length=min_len, do_sample=False)
        if not result:
            print("safe_summarize: Summarizer returned empty result list.")
            return ""
        if 'summary_text' not in result[0]:
            print(f"safe_summarize: 'summary_text' missing in summarizer output: {result}")
            return ""
        return result[0]['summary_text']
    except Exception as e:
        print(f"Summarizer error: {e}")
    return ""

def split_text(text, max_words=500):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def summarize_text_short(text):
    chunks = split_text(text, max_words=500)
    summaries = []
    for chunk in chunks:
        summary = safe_summarize(chunk, max_len=100, min_len=30)
        if summary:
            summaries.append(summary)
    return "\n\n".join(summaries)

def summarize_text_medium(text):
    chunks = split_text(text, max_words=500)
    summaries = []
    for chunk in chunks:
        summary = safe_summarize(chunk, max_len=250, min_len=80)
        if summary:
            summaries.append(summary)
    return "\n\n".join(summaries)

def summarize_text_long(text):
    chunks = split_text(text, max_words=500)
    summaries = []
    for chunk in chunks:
        summary = safe_summarize(chunk, max_len=400, min_len=150)
        if summary:
            summaries.append(summary)
    return "\n\n".join(summaries)

def summarize_text_formal(text):
    prompt_prefix = "Please provide a formal summary:\n\n"
    max_chunk_words = 400  # smaller chunk size for prompt+text
    chunks = split_text(text, max_words=max_chunk_words)
    summaries = []
    for chunk in chunks:
        prompt = prompt_prefix + chunk
        summary = safe_summarize(prompt, max_len=250, min_len=80)
        if summary:
            summaries.append(summary)
    return "\n\n".join(summaries)


def summarize_text_simplified(text):
    prompt_prefix = "Summarize the following text in simple language:\n\n"
    max_chunk_words = 400  # reduce chunk size if needed

    chunks = split_text(text, max_words=max_chunk_words)
    summaries = []

    for chunk in chunks:
        prompt = prompt_prefix + chunk
        summary = safe_summarize(prompt, max_len=200, min_len=80)
        if summary:
            summaries.append(summary)

    return "\n\n".join(summaries)

@app.route("/")
def home():
    return render_template("index.html")


def creative_summarize(text, summary_type="medium", max_len=200, min_len=80):
    try:
        if summary_type in ["formal", "simple"]:
            # Use FLAN for instruction-style prompts
            if summary_type == "formal":
                prompt = f"Please provide a formal summary:\n\n{text}"
            elif summary_type == "simple":
                prompt = f"Summarize the following text in simple language:\n\n{text}"

            result = summarizer_prompt(
                prompt,
                max_length=max_len,
                min_length=min_len,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
            )
            if result and 'generated_text' in result[0]:
                return result[0]['generated_text']

        else:
            # Use distilbart for short, medium, long (no prompt)
            length_config = {
                "short": (100, 30),
                "medium": (250, 80),
                "long": (400, 150)
            }
            max_l, min_l = length_config.get(summary_type, (250, 80))
            result = summarizer_standard(
                text,
                max_length=max_l,
                min_length=min_l,
                do_sample=False
            )
            if result and 'summary_text' in result[0]:
                return result[0]['summary_text']

    except Exception as e:
        print(f"❌ creative_summarize error: {e}")
    return ""


@app.route('/resummarize', methods=['POST'])
def resummarize():
    try:
        data = request.get_json()
        raw_url = data.get("url")
        summary_type = data.get("summary_type", "summary_medium").replace("summary_", "")
        user_id = data.get("user_id", "anonymous")

        if not raw_url:
            return jsonify({"error": "Missing YouTube URL"}), 400

        url = clean_youtube_url(raw_url)

        # ✅ Prefer subtitles over transcription
        if has_subtitles(url):
            subs_path = download_youtube_subtitles(url)
            combined_text = read_subtitle_text(subs_path)
        else:
            audio_path = download_youtube_audio(url)
            combined_text = transcribe_audio(audio_path)

        # ✅ Validate transcript
        if not combined_text or len(combined_text.strip()) < 50:
            return jsonify({"error": "Transcript is too short or empty."}), 400

        # ✅ Only generate the requested summary type using the proper model
        summary_lengths = {
            "short": (100, 30),
            "medium": (250, 80),
            "long": (400, 150),
            "formal": (250, 80),
            "simple": (200, 80)
        }

        if summary_type not in summary_lengths:
            return jsonify({"error": f"Invalid summary type: {summary_type}"}), 400

        max_len, min_len = summary_lengths[summary_type]
        new_summary = creative_summarize(
            text=combined_text,
            summary_type=summary_type,
            max_len=max_len,
            min_len=min_len
        )

        # ✅ Store feedback (neutral for now)
        save_feedback(user_id, url, new_summary, 0)

        return jsonify({
            "new_summary": new_summary,
            "variant": f"summary_{summary_type}"
        })

    except Exception as e:
        print("🔥 Error (resummarize):", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
