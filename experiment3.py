"""
Experiment 3: Annotation Failure Analysis
Analyzes where in the state-tracking CoT process failures occur.

For each failed state_tracking_cot response it checks:
- Were the intermediate [State: X] annotations correct at each step?
- Did the model track correctly but give the wrong final answer?
- Or did the tracking itself go wrong first?
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


def generate_sequence(n_events, write_prob, seed=None):
    if seed is not None:
        random.seed(seed)
    events = []
    states = []
    current_state = None
    for i in range(n_events):
        is_write = True if i == 0 else random.random() < write_prob
        if is_write:
            val = random.choice(VALUES)
            events.append(random.choice(WRITE_TEMPLATES).format(val=val))
            current_state = val
        else:
            events.append(random.choice(DISTRACTORS))
        states.append(current_state)
    if current_state is None:
        val = random.choice(VALUES)
        events[0] = random.choice(WRITE_TEMPLATES).format(val=val)
        current_state = val
        states[0] = val
    sequence_text = "\n".join(f"{i+1}. {e}" for i, e in enumerate(events))
    return sequence_text, current_state, events, states


def get_seed(model, density, n_events, condition, sample_idx):
    return hash((model, density, n_events, condition, sample_idx)) % (2**31)


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
                max_tokens=1200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return ""


def build_state_tracking_prompt(sequence):
    system = (
        "You are tracking the state of a switch. "
        "Read each event one by one. After each event, write the current switch state "
        "in brackets like [State: ON] or [State: OFF]. "
        "If the event doesn't change the switch, keep the same state. "
        "At the end, output your final answer on a new line as: Answer: ON or Answer: OFF"
    )
    user = f"{sequence}\n\nQuestion: What is the switch currently set to? Process each event and track the state."
    return system, user


def extract_annotations(output):
    matches = re.findall(r'\[State:\s*(ON|OFF)\]', output, re.IGNORECASE)
    return [m.upper() for m in matches]


def extract_final_answer(output):
    match = re.search(r'Answer:\s*(ON|OFF)', output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    matches = re.findall(r'\b(ON|OFF)\b', output.upper())
    return matches[-1] if matches else None


def analyze_failure(output, gold, ground_truth_states, events):
    annotations = extract_annotations(output)
    final_answer = extract_final_answer(output)

    if final_answer is None:
        return {"failure_type": "no_answer", "final_answer": None,
                "annotations": annotations, "first_wrong_annotation": None}

    correct_final = (final_answer == gold)

    if correct_final:
        return {"failure_type": "correct", "final_answer": final_answer,
                "annotations": annotations, "first_wrong_annotation": None}

    # Find where tracking went wrong, compare annotations
    first_wrong = None
    n_compare = min(len(annotations), len(ground_truth_states))
    for i in range(n_compare):
        if annotations[i] != ground_truth_states[i]:
            first_wrong = i
            break

    if first_wrong is not None:
        # Tracking went wrong before the answer
        failure_type = "tracking_error"
    else:
        # Tracking was correct (or maybe we ran out of annotations to check), but final answer is still wrong
        failure_type = "answer_error"

    return {
        "failure_type": failure_type,
        "final_answer": final_answer,
        "gold": gold,
        "annotations": annotations,
        "n_annotations": len(annotations),
        "n_events": len(events),
        "first_wrong_annotation": first_wrong,
        "ground_truth_states": ground_truth_states,
    }


def run_experiment3(results1_file="results.json", output_file="results3.json"):
    with open(results1_file) as f:
        results1 = json.load(f)

    density_probs = {"sparse": 0.15, "dense": 0.60}

    target_rows = [
        r for r in results1
        if r["model"] == "gpt-3.5-turbo"
        and r["condition"] == "state_tracking_cot"
        and r["accuracy"] < 1.0
    ]

    all_results = []
    failure_analyses = []

    for row in target_rows:
        density = row["density"]
        n_events = row["n_events"]
        write_prob = density_probs[density]

        tracking_errors = 0
        answer_errors = 0
        no_answers = 0
        correct_count = 0
        failure_cases = []

        for sample_idx in range(100):
            seed = get_seed("gpt-3.5-turbo", density, n_events, "state_tracking_cot", sample_idx)
            sequence, gold, events, states = generate_sequence(
                n_events, write_prob, seed=seed
            )

            system, user = build_state_tracking_prompt(sequence)
            output = query(system, user, model="gpt-3.5-turbo")

            analysis = analyze_failure(output, gold, states, events)

            if analysis["failure_type"] == "correct":
                correct_count += 1
            elif analysis["failure_type"] == "tracking_error":
                tracking_errors += 1
                failure_cases.append({
                    "sample_idx": sample_idx,
                    "output_snippet": output[:500],
                    **analysis
                })
                print(f"  Tracking error sample {sample_idx}: "
                      f"first wrong annotation at position {analysis['first_wrong_annotation']}")
            elif analysis["failure_type"] == "answer_error":
                answer_errors += 1
                failure_cases.append({
                    "sample_idx": sample_idx,
                    "output_snippet": output[:500],
                    **analysis
                })
                print(f"  Answer error sample {sample_idx}: "
                      f"tracked correctly but answered {analysis['final_answer']} (gold={gold})")
            else:
                no_answers += 1
                print(f"  No answer sample {sample_idx}")

        total_failures = tracking_errors + answer_errors + no_answers
        print(f"  Correct: {correct_count} | Tracking errors: {tracking_errors} | "
              f"Answer errors: {answer_errors} | No answer: {no_answers}")

        result = {
            "model": "gpt-3.5-turbo",
            "density": density,
            "n_events": n_events,
            "original_accuracy": row["accuracy"],
            "correct": correct_count,
            "tracking_errors": tracking_errors,
            "answer_errors": answer_errors,
            "no_answers": no_answers,
            "total_failures": total_failures,
            "pct_tracking": tracking_errors / total_failures if total_failures > 0 else 0,
            "pct_answer": answer_errors / total_failures if total_failures > 0 else 0,
            "failure_cases": failure_cases,
        }
        all_results.append(result)

        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_experiment3()
