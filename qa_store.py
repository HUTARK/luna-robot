import json
import os

# Self-locating: the old hardcoded /home/groot/... breaks the moment this is
# cloned under a different username, and silently — load_qa() just returns {}
# and the robot answers nothing from its local store.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QA_FILE = os.path.join(BASE_DIR, "qa_store.json")


def load_qa():
    if os.path.exists(QA_FILE):
        try:
            with open(QA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[QA] could not read {QA_FILE}: {e}")
    return {}


def save_qa(qa):
    tmp = QA_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(qa, f, indent=2)
    os.replace(tmp, QA_FILE)      # atomic: never leaves a half-written store


def find_answer(question: str):
    """Match a question against the stored Q&As (id-keyed format).

    Both sides are lowercased. The previous version lowercased only the
    incoming question and compared it to the stored text as-is, so any entry
    saved with a capital letter — "What is your name" — could never match.
    Since the app lets people type questions naturally, most entries had one.
    """
    if not question:
        return None
    q_in = question.lower().strip()
    qa = load_qa()

    # Exact match first.
    for entry in qa.values():
        if isinstance(entry, dict):
            if entry.get('q', '').lower().strip() == q_in:
                return entry.get('a')

    # Then substring either way, longest stored question first so a specific
    # entry wins over a short generic one. Without the sort, "weather" would
    # shadow "weather in Chennai tomorrow" depending on dict order.
    candidates = []
    for entry in qa.values():
        if not isinstance(entry, dict):
            continue
        q = entry.get('q', '').lower().strip()
        if not q:
            continue
        if q in q_in or q_in in q:
            candidates.append((len(q), entry.get('a')))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    return None
