"""
Experiment 4: Error Consistency Check
Re-runs sequences that failed under direct prompting 3 more times
Tests whether errors are sporadic (random) or systematic (same sequence always fails)
"""

import openai
import json
import random
import re
import time

client = openai.OpenAI()

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

RERUNS = 3  # how many times to re-run each failing sequence


def generate_sequence(n_events, write_prob, seed=None):
    if seed is not None:
        random.seed(seed)
    events = []
    current_state = None
    for i in range(n_events):
        is_write = True if i == 0 else random.random() < write_prob
        if is_write:
            val = random.choice(VALUES)
            events.append(random.choice(WRITE_TEMPLATES).format(val=val))
            current_state = val
        else:
            events.append(random.choice(DISTRACTORS))
    if current_state is None:
        val = random.choice(VALUES)
        events[0] = random.choice(WRITE_TEMPLATES).format(val=val)
        current_state = val
    return "\n".join(f"{i+1}. {e}" for i, e in enumerate(events)), current_state


def parse_direct(output):
    clean = output.strip().upper()
    if clean in ("ON", "OFF"):
        return clean
    match = re.search(r'\b(ON|OFF)\b', output.upper())
    return match.group(1) if match else None


def query(system, user, model="gpt-3.5-turbo", retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error: {e}")
                return ""


def get_seed(model, density, n_events, condition, sample_idx):
    return hash((model, density, n_events, condition, sample_idx)) % (2**31)


def run_experiment4(
    results1_file="results.json",
    output_file="results4.json"
):
    # Load experiment 1 results to find which sequences failed
    with open(results1_file) as f:
        results1 = json.load(f)

    target_conditions = [
        r for r in results1
        if r["model"] == "gpt-3.5-turbo"
        and r["condition"] == "direct"
        and r["n_events"] in [30, 50]
    ]

    system = (
        "You are tracking the state of a switch. "
        "Read the sequence of events carefully and answer the final question. "
        "Output ONLY 'ON' or 'OFF' with no explanation."
    )

    results = []
    density_probs = {"sparse": 0.15, "dense": 0.60}

    for cond_row in target_conditions:
        density = cond_row["density"]
        n_events = cond_row["n_events"]
        write_prob = density_probs[density]

        sequence_results = []

        for sample_idx in range(100):
            seed = get_seed("gpt-3.5-turbo", density, n_events, "direct", sample_idx)
            sequence, gold = generate_sequence(n_events, write_prob, seed=seed)

            # Run once to check if this specific sequence fails
            user = f"{sequence}\n\nQuestion: What is the switch currently set to?"
            output1 = query(system, user)
            pred1 = parse_direct(output1)
            first_correct = (pred1 == gold) if pred1 else False

            if not first_correct:
                rerun_results = []
                for _ in range(RERUNS):
                    output = query(system, user)
                    pred = parse_direct(output)
                    rerun_results.append((pred == gold) if pred else False)

                n_correct_reruns = sum(rerun_results)
                # Consistent error: fails on reruns too -> systematic
                # Sporadic error: sometimes passes on reruns -> random
                consistent = n_correct_reruns == 0
                sequence_results.append({
                    "sample_idx": sample_idx,
                    "gold": gold,
                    "first_pred": pred1,
                    "first_correct": first_correct,
                    "rerun_correct": rerun_results,
                    "n_correct_reruns": n_correct_reruns,
                    "consistent_error": consistent,
                })
                status = "CONSISTENT" if consistent else f"SPORADIC ({n_correct_reruns}/{RERUNS} reruns correct)"

        # Summarize the results
        n_failures = len(sequence_results)
        n_consistent = sum(1 for r in sequence_results if r["consistent_error"])
        n_sporadic = n_failures - n_consistent
        sporadic_rate = n_sporadic / n_failures if n_failures > 0 else 0

        print(f"  Total failures: {n_failures}")
        print(f"  Consistent errors (systematic): {n_consistent} ({n_consistent/n_failures*100:.1f}%)" if n_failures > 0 else "  No failures")
        print(f"  Sporadic errors (random): {n_sporadic} ({sporadic_rate*100:.1f}%)" if n_failures > 0 else "")

        results.append({
            "model": "gpt-3.5-turbo",
            "density": density,
            "n_events": n_events,
            "original_accuracy": cond_row["accuracy"],
            "n_failures_found": n_failures,
            "n_consistent_errors": n_consistent,
            "n_sporadic_errors": n_sporadic,
            "sporadic_rate": sporadic_rate,
            "sequence_details": sequence_results,
        })

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_experiment4()
