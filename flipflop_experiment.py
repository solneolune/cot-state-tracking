"""
Flip-Flop State Tracking Experiment
"""

import openai
import json
import random
import re
import time
from dataclasses import dataclass, asdict

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


# Sequence generator

def generate_sequence(n_events: int, write_prob: float, seed: int = None):
    if seed is not None:
        random.seed(seed)
    
    events = []
    current_state = None
    n_writes = 0

    for i in range(n_events):
        # Force at least one write somewhere in first third
        if i == 0:
            is_write = True
        else:
            is_write = random.random() < write_prob

        if is_write:
            val = random.choice(VALUES)
            template = random.choice(WRITE_TEMPLATES)
            events.append(template.format(val=val))
            current_state = val
            n_writes += 1
        else:
            events.append(random.choice(DISTRACTORS))

    # make sure that there's at least one write
    if current_state is None:
        val = random.choice(VALUES)
        events[0] = random.choice(WRITE_TEMPLATES).format(val=val)
        current_state = val
        n_writes = 1

    sequence_text = "\n".join(f"{i+1}. {e}" for i, e in enumerate(events))
    return sequence_text, current_state, n_writes


# Prompt builders

def build_prompt_direct(sequence: str) -> tuple[str, str]:
    system = (
        "You are tracking the state of a switch. "
        "Read the sequence of events carefully and answer the final question. "
        "Output ONLY 'ON' or 'OFF' with no explanation."
    )
    user = f"{sequence}\n\nQuestion: What is the switch currently set to?"
    return system, user


def build_prompt_generic_cot(sequence: str) -> tuple[str, str]:
    system = (
        "You are tracking the state of a switch. "
        "Read the sequence of events carefully. "
        "Think step by step before answering. "
        "At the end, output your final answer on a new line as: Answer: ON or Answer: OFF"
    )
    user = f"{sequence}\n\nQuestion: What is the switch currently set to?"
    return system, user


def build_prompt_state_tracking_cot(sequence: str) -> tuple[str, str]:
    system = (
        "You are tracking the state of a switch. "
        "Read each event one by one. After each event, write the current switch state in brackets like [State: ON] or [State: OFF]. "
        "If the event doesn't change the switch, keep the same state. "
        "At the end, output your final answer on a new line as: Answer: ON or Answer: OFF"
    )
    user = f"{sequence}\n\nQuestion: What is the switch currently set to? Process each event and track the state."
    return system, user


PROMPT_BUILDERS = {
    "direct": build_prompt_direct,
    "generic_cot": build_prompt_generic_cot,
    "state_tracking_cot": build_prompt_state_tracking_cot,
}


# Answer parser

def parse_answer(output: str, condition: str) -> str | None:
    if condition == "direct":
        # Should be just ON or OFF
        clean = output.strip().upper()
        if clean in ("ON", "OFF"):
            return clean
        match = re.search(r'\b(ON|OFF)\b', output.upper())
        return match.group(1) if match else None
    else:
        # search for "Answer: ON/OFF"
        match = re.search(r'Answer:\s*(ON|OFF)', output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        matches = re.findall(r'\b(ON|OFF)\b', output.upper())
        return matches[-1] if matches else None


def count_state_annotations(output: str) -> int:
    return len(re.findall(r'\[State:\s*(ON|OFF)\]', output, re.IGNORECASE))


# API call

def query_model(system: str, user: str, model: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error after {retries} attempts: {e}")
                return ""


# Experiment configs

@dataclass
class ExperimentConfig:
    n_samples: int = 100
    models: tuple = ("gpt-3.5-turbo", "gpt-4o-mini")
    conditions: tuple = ("direct", "generic_cot", "state_tracking_cot")
    n_events_levels: tuple = (5, 15, 30, 50)
    density_levels: dict = None

    def __post_init__(self):
        if self.density_levels is None:
            self.density_levels = {
                "sparse": 0.15,
                "dense": 0.60,
            }


# Main experiment loop

def run_experiment(config: ExperimentConfig, output_file: str = "results.json"):
    results = []

    total = (config.n_samples * len(config.models) * len(config.conditions) *
             len(config.n_events_levels) * len(config.density_levels))
    done = 0

    for model in config.models:
        for density_name, write_prob in config.density_levels.items():
            for n_events in config.n_events_levels:
                for condition in config.conditions:

                    correct = 0
                    parse_failures = 0
                    annotation_counts = []

                    for sample_idx in range(config.n_samples):
                        # Use deterministic seed so experiment is reproducible
                        seed = hash((model, density_name, n_events, condition, sample_idx)) % (2**31)
                        sequence, gold, n_writes = generate_sequence(
                            n_events=n_events,
                            write_prob=write_prob,
                            seed=seed
                        )

                        system, user = PROMPT_BUILDERS[condition](sequence)
                        output = query_model(system, user, model)
                        pred = parse_answer(output, condition)

                        if pred is None:
                            parse_failures += 1
                        elif pred == gold:
                            correct += 1

                        if condition == "state_tracking_cot":
                            annotation_counts.append(count_state_annotations(output))

                    accuracy = correct / config.n_samples
                    glitch_rate = 1 - accuracy

                    result = {
                        "model": model,
                        "density": density_name,
                        "n_events": n_events,
                        "condition": condition,
                        "accuracy": accuracy,
                        "glitch_rate": glitch_rate,
                        "parse_failures": parse_failures,
                        "mean_annotations": (
                            sum(annotation_counts) / len(annotation_counts)
                            if annotation_counts else None
                        ),
                        "n_samples": config.n_samples,
                    }
                    results.append(result)

                    done += config.n_samples
                    progress = done / total * 100
                    print(
                        f"[{progress:.1f}%] {model} | {density_name} | "
                        f"n={n_events} | {condition} → "
                        f"acc={accuracy:.2f} glitch={glitch_rate:.2f} "
                        f"parse_fail={parse_failures}"
                    )

                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2)

    return results


#  Quick check on 5 sample on n=15 before running the full experiment

def sanity_check():
    for condition in ("direct", "generic_cot", "state_tracking_cot"):
        print(f"\n {condition}")
        for i in range(3):
            seq, gold, _ = generate_sequence(n_events=15, write_prob=0.15, seed=i)
            system, user = PROMPT_BUILDERS[condition](seq)
            output = query_model(system, user, "gpt-3.5-turbo")
            pred = parse_answer(output, condition)
            status = "v" if pred == gold else "x"
            print(f"  [{status}] gold={gold} pred={pred}")
            if i == 0:
                print(f"  Output snippet: {output[:200]}...")


# Entry
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sanity":
        sanity_check()
    else:
        config = ExperimentConfig(n_samples=100)
        run_experiment(config, output_file="results.json")
