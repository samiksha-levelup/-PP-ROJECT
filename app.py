from flask import Flask, request, jsonify, render_template

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

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


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
        'noplaylist': True
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

    # Extract basic math expressions like a + b, x^2, etc.
    symbol_formulas = re.findall(r'\b[a-zA-Z0-9]+\s*[\+\-\*/\^]\s*[a-zA-Z0-9]+\b', text)
    formulas.update(symbol_formulas)

    # Extract written fractions but ignore unhelpful ones like 1 over 1, 3 over 3
    written_fractions = re.findall(r'\b(\d+)\s+over\s+(\d+)\b', text)
    for num, den in written_fractions:
        if num != den:  # ignore 1 over 1, 3 over 3, etc.
            formulas.add(f"{num} over {den}")

    # Add support for "x squared", "x cubed"
    powers = re.findall(r'\b([a-zA-Z])\s+(squared|cubed)\b', text)
    for var, power in powers:
        if power == "squared":
            formulas.add(f"{var}^2")
        elif power == "cubed":
            formulas.add(f"{var}^3")

    return sorted(formulas, key=len, reverse=True)


import re

def clean_text(text):
    """Normalize spaces, lowercase, remove punctuation except math symbols."""
    text = re.sub(r"\s+", " ", text)              # Collapse multiple spaces
    text = re.sub(r"[^\w\s=+\-*/^0-9.]", "", text) # Remove most punctuation
    return text.lower().strip()
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10"
}
from word2number import w2n

def convert_word_math_to_symbols(text):
    text = text.lower()

    # Replace phrases like "is equal to", "equals" with "=" first
    text = re.sub(r'\bis equal to\b|\bequals\b', '=', text)

    # Replace math operators
    text = re.sub(r'\bplus\b', '+', text)
    text = re.sub(r'\bminus\b', '-', text)
    text = re.sub(r'\btimes\b|\bmultiplied by\b', '*', text)
    text = re.sub(r'\bdivided by\b', '/', text)

    # Convert number words to digits using word2number
    def replace_number_words(match):
        try:
            return str(w2n.word_to_num(match.group(0)))
        except:
            return match.group(0)

    # Match potential number phrases like "twenty five"
    text = re.sub(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                  r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                  r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                  r'eighty|ninety|hundred|thousand)(?:[\s-](?:one|two|three|four|five|six|seven|eight|nine))?\b',
                  replace_number_words, text)

    return text



def extract_examples(text):
    math_phrase_replacements = {
        r'\bone\s+plus\s+one\s+plus\s+one\b': '1 + 1 + 1',
        r'\bone\s+plus\s+one\b': '1 + 1',
        r'\btwo\s+minus\s+one\b': '2 - 1',
        r'\bthree\s+minus\s+two\b': '3 - 2',
        r'\bfour\s+plus\s+four\b': '4 + 4',
        r'\bbinary\s+equivalent\s+of\s+3\s+is\s+1\s+1\b': 'Binary of 3 = 11',
        r'\bis\s+equal\s+to\b': '=',
        r'\bit\s+is\s+equal\s+to\b': '=',
        r'\bequals\b': '=',
    }

    for pattern, replacement in math_phrase_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    trigger_phrases = r"(?:for example|say we have|let's say|suppose|consider|imagine|assume)"
    pattern = rf"({trigger_phrases}[^\.!?]*[\.!?](?:\s[^\.!?]*[\.!?])?)"
    matches = re.findall(pattern, text, re.IGNORECASE)

    seen = set()
    examples = []
    for match in matches:
        match = match.strip()
        match_cleaned = clean_text(match)
        if match_cleaned not in seen:
            seen.add(match_cleaned)
            converted = convert_word_math_to_symbols(match)
            examples.append(converted)


    return examples




from flask import request, jsonify

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()

    user_id = data.get('user_id', 'anonymous')
    video_url = data.get('video_url', '')
    summary = data.get('summary', '')  # Use .get() to avoid KeyError
    user_feedback = data.get('feedback')

    # Log for debugging
    print(f"User: {user_id}, Video: {video_url}, Feedback: {user_feedback}, Summary: {summary[:30]}...")

    # Save to SQLite here
    try:
        save_feedback(user_id, video_url, summary, user_feedback)
    except Exception as e:
        print(f"Error saving feedback to DB: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "success"})

summary_stats = {
    "variant_1": {"count": 0, "total_reward": 0},
    "variant_2": {"count": 0, "total_reward": 0},
    # etc
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
    stat = summary_stats[variant]
    stat["count"] += 1
    stat["total_reward"] += reward


def generate_summary_variants(text, formulas):
    variants = {}
    for name, max_len in [("short", 100), ("medium", 250), ("long", 400)]:
        chunks = split_text(text, max_words=500)
        summaries = []
        for chunk in chunks:
            replaced = replace_formulas_with_placeholders(chunk, formulas)
            summary_result = summarizer(replaced, max_length=max_len, min_length=50, do_sample=False)
            summary_text = summary_result[0]['summary_text']
            final_summary = reinject_formulas(summary_text, formulas)
            summaries.append(final_summary)
        variants[name] = "\n\n".join(summaries)
    return variants








def replace_formulas_with_placeholders(text, formulas):
    for i, f in enumerate(formulas):
        placeholder = f"[FORMULA_{i}]"
        text = re.sub(re.escape(f), placeholder, text)
    return text

def reinject_formulas(text, formulas):
    for i, f in enumerate(formulas):
        placeholder = f"[FORMULA_{i}]"
        converted = words_to_math_symbols(f)
        text = text.replace(placeholder, f"\\({converted}\\)")
    return text




def split_text(text, max_words=500):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def summarize_large_text_with_formulas(text, formulas):
    chunks = split_text(text, max_words=500)
    summaries = []
    for chunk in chunks:
        replaced = replace_formulas_with_placeholders(chunk, formulas)
        summary_result = summarizer(replaced, max_length=250, min_length=100, do_sample=False)
        summary_text = summary_result[0]['summary_text']
        final_summary = reinject_formulas(summary_text, formulas)
        summaries.append(final_summary)
    return "\n\n".join(summaries)

# ----------- Routes -----------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/youtube", methods=["POST"])
def youtube_summary():
    try:
        data = request.get_json()
        raw_url = data.get("url")
        if not raw_url:
            return jsonify({"error": "No URL provided"}), 400

        url = clean_youtube_url(raw_url)

        # ✅ Prefer subtitles over transcription if available
        if has_subtitles(url):
            print("✅ Using subtitles instead of transcription.")
            subs_path = download_youtube_subtitles(url)
            combined_text = read_subtitle_text(subs_path)
        else:
            print("🎧 Subtitles not available — falling back to audio transcription.")
            audio_path = download_youtube_audio(url)
            transcript = transcribe_audio(audio_path)
            combined_text = transcript

        # Extract formulas and examples
        formulas = extract_formulas(combined_text)
        examples = extract_examples(combined_text)

        # Summarize using combined text and formulas
        summary = summarize_large_text_with_formulas(combined_text, formulas)

        return jsonify({
            "summary": summary,
            "formulas": [f"\\({words_to_math_symbols(f)}\\)" for f in formulas],
            "examples": examples
        })

    except Exception as e:
        print("🔥 Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
