"""
Experiment 2: Self-Consistency and Reverse Reading
Tests two additional mitigation strategies on the hardest conditions only
"""

import openai
import json
import random
import re
import time
from collections import Counter

client = openai.OpenAI()

# Reuse generators from experiment 1

DISTRACTORS = [
    "Bob eats a sandwich.",
    "Alice reads a book.",
    "Bob goes for a walk.",
    "Alice drinks coffee.",
    "Bob checks his phone.",
    "Alice opens a window.",
    "Bob puts on his coat.",
    "Alice looks outside.",
    "Bob hums to himself.",
    "Alice stretches her arms.",
    "Bob picks up a pen.",
    "Alice clears her throat.",
    "Bob taps the table.",
    "Alice glances at the clock.",
    "Bob sighs quietly.",
    "Alice adjusts her glasses.",
    "Bob flips through a magazine.",
    "Alice pours a glass of water.",
]

WRITE_TEMPLATES = [
    "Alice sets the switch to {val}.",
    "Alice flips the switch to {val}.",
    "Alice turns the switch {val}.",
    "Alice changes the switch to {val}.",
]

VALUES = ["ON", "OFF"]


def generate_sequence(n_events, write_prob, seed=None):
    if seed is not None:
        random.seed(seed)
    events = []
    current_state = None
    n_writes = 0
    for i in range(n_events):
        is_write = True if i == 0 else random.random() < write_prob
        if is_write:
            val = random.choice(VALUES)
            events.append(random.choice(WRITE_TEMPLATES).format(val=val))
            current_state = val
            n_writes += 1
        else:
            events.append(random.choice(DISTRACTORS))
    if current_state is None:
        val = random.choice(VALUES)
        events[0] = random.choice(WRITE_TEMPLATES).format(val=val)
        current_state = val
    sequence_text = "\n".join(f"{i+1}. {e}" for i, e in enumerate(events))
    return sequence_text, current_state, n_writes


# Prompt builders

def build_self_consistency(sequence):
    system = (
        "You are tracking the state of a switch. "
        "Read the sequence of events carefully and answer the final question. "
        "Output ONLY 'ON' or 'OFF' with no explanation."
    )
    user = f"{sequence}\n\nQuestion: What is the switch currently set to?"
    return system, user


def build_reverse_reading(sequence):
    system = (
        "You are tracking the state of a switch. "
        "To answer the question, scan the events from LAST to FIRST. "
        "Find the most recent event where Alice changes the switch. "
        "That event gives you the answer. "
        "Show your backward scan briefly, then output your answer as: Answer: ON or Answer: OFF"
    )
    user = f"{sequence}\n\nQuestion: What is the switch currently set to? Scan backwards to find the answer."
    return system, user


# Answer parser
def parse_direct(output):
    clean = output.strip().upper()
    if clean in ("ON", "OFF"):
        return clean
    match = re.search(r'\b(ON|OFF)\b', output.upper())
    return match.group(1) if match else None


def parse_cot(output):
    match = re.search(r'Answer:\s*(ON|OFF)', output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    matches = re.findall(r'\b(ON|OFF)\b', output.upper())
    return matches[-1] if matches else None


# API call
def query(system, user, model="gpt-3.5-turbo", temperature=0, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error: {e}")
                return ""


#Self-consistency
SC_SAMPLES = 5

def run_self_consistency(sequence, gold, model="gpt-3.5-turbo"):
    system, user = build_self_consistency(sequence)
    votes = []
    for _ in range(SC_SAMPLES):
        # temperature = 0.7 for diverse samples
        output = query(system, user, model=model, temperature=0.7)
        pred = parse_direct(output)
        if pred is not None:
            votes.append(pred)
    if not votes:
        return None, votes
    majority = Counter(votes).most_common(1)[0][0]
    return majority, votes


# Main experiment
MODEL = "gpt-3.5-turbo"
N_EVENTS_LEVELS = [30, 50]
DENSITY_LEVELS = {"sparse": 0.15, "dense": 0.60}
N_SAMPLES = 100

# Use the same seeds as in experiment 1 so that the sequences are identical to make direct comparison valid
def get_seed(model, density, n_events, condition, sample_idx):
    return hash((model, density, n_events, condition, sample_idx)) % (2**31)


def run_experiment2(output_file="results2.json"):
    results = []
    conditions = ["self_consistency", "reverse_reading"]

    total = N_SAMPLES * len(N_EVENTS_LEVELS) * len(DENSITY_LEVELS) * len(conditions)
    total_calls = (N_SAMPLES * len(N_EVENTS_LEVELS) * len(DENSITY_LEVELS) * SC_SAMPLES +
                   N_SAMPLES * len(N_EVENTS_LEVELS) * len(DENSITY_LEVELS))
    done = 0

    for density_name, write_prob in DENSITY_LEVELS.items():
        for n_events in N_EVENTS_LEVELS:
            for condition in conditions:
                correct = 0
                parse_failures = 0
                vote_details = []  # for self-consistency analysis

                for sample_idx in range(N_SAMPLES):
                    seed = get_seed(MODEL, density_name, n_events, "direct", sample_idx)
                    sequence, gold, _ = generate_sequence(
                        n_events=n_events,
                        write_prob=write_prob,
                        seed=seed
                    )

                    if condition == "self_consistency":
                        pred, votes = run_self_consistency(sequence, gold, model=MODEL)
                        vote_details.append({
                            "gold": gold,
                            "pred": pred,
                            "votes": votes,
                            "correct": pred == gold if pred else False
                        })
                    else:
                        system, user = build_reverse_reading(sequence)
                        output = query(system, user, model=MODEL)
                        pred = parse_cot(output)

                    if pred is None:
                        parse_failures += 1
                    elif pred == gold:
                        correct += 1

                accuracy = correct / N_SAMPLES
                result = {
                    "model": MODEL,
                    "density": density_name,
                    "n_events": n_events,
                    "condition": condition,
                    "accuracy": accuracy,
                    "glitch_rate": 1 - accuracy,
                    "parse_failures": parse_failures,
                    "n_samples": N_SAMPLES,
                }
                if condition == "self_consistency":
                    confident = sum(
                        1 for v in vote_details
                        if v["votes"] and Counter(v["votes"]).most_common(1)[0][1] >= 4
                    )
                    result["sc_confident_rate"] = confident / N_SAMPLES
                    result["vote_details"] = vote_details

                results.append(result)
                done += N_SAMPLES

                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run_experiment2()
