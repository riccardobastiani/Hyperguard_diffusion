import json
import csv
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate jailbreak dataset from eval results.")
    parser.add_argument('--input', required=True, help='Path to the eval results JSON file')
    parser.add_argument('--output_json', required=True, help='Path for the output JSON file')
    parser.add_argument('--output_csv', required=True, help='Path for the output CSV file')
    parser.add_argument(
        '--filter',
        choices=['asr_e', 'asr_k', 'all'],
        default='asr_e',
        help='Which rows to keep: asr_e for harmful jailbreak datasets, asr_k for non-refusal, all for harmless datasets.'
    )
    return parser.parse_args()

def is_positive_label(value):
    return value is True or value == 1 or (isinstance(value, float) and value >= 0.5)

def should_keep(entry, filter_mode, invert_asr_e=False):
    if filter_mode == 'all':
        return True
    if filter_mode == 'asr_k':
        return is_positive_label(entry.get('asr_k_label', 0))
    asr_e_value = is_positive_label(entry.get('asr_e_label', 0))
    return (not asr_e_value) if invert_asr_e else asr_e_value

def is_run_alpaca_path(input_file):
    parts = os.path.normpath(input_file).lower().split(os.sep)
    return 'run_alpaca' in parts

def process_jailbreaks(input_file, file_output_json, file_output_csv, filter_mode='asr_e'):
    # Carica i dati dal file dei risultati di valutazione
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    successful_jailbreaks = []

    # Support both dict-of-lists (HarmBench) and flat list (StrongREJECT) formats
    if isinstance(data, dict):
        entries = [entry for results in data.values() for entry in results]
    else:
        entries = data

    invert_asr_e = filter_mode == 'asr_e' and is_run_alpaca_path(input_file)

    for entry in entries:
        if should_keep(entry, filter_mode, invert_asr_e=invert_asr_e):
            response_text = entry.get('generation') or entry.get('response', '')

            record = {
                'Behavior': entry.get('Behavior') or entry.get('behavior') or entry.get('goal') or entry.get('vanilla prompt', ''),
                'Refined_behavior': entry.get('Refined_behavior') or entry.get('refined goal') or entry.get('refine prompt', ''),
                'Response': response_text
            }
            successful_jailbreaks.append(record)

    # --- Salvataggio in JSON ---
    os.makedirs(os.path.dirname(file_output_json), exist_ok=True)
    with open(file_output_json, 'w', encoding='utf-8') as f_json:
        json.dump(successful_jailbreaks, f_json, indent=4, ensure_ascii=False)
    print(f"File JSON creato con successo: {file_output_json} ({len(successful_jailbreaks)} record)")

    # --- Salvataggio in CSV ---
    keys = ['Behavior', 'Refined_behavior', 'Response']
    with open(file_output_csv, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=keys)
        writer.writeheader()
        writer.writerows(successful_jailbreaks)
    print(f"File CSV creato con successo: {file_output_csv}")

if __name__ == "__main__":
    args = parse_args()
    process_jailbreaks(args.input, args.output_json, args.output_csv, args.filter)
