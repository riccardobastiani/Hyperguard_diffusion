import json
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate jailbreak dataset from eval results.")
    parser.add_argument('--input', required=True, help='Path to the eval results JSON file')
    parser.add_argument('--output_json', required=True, help='Path for the output JSON file')
    parser.add_argument('--output_csv', required=True, help='Path for the output CSV file')
    return parser.parse_args()

def process_jailbreaks(input_file, file_output_json, file_output_csv):
    # Carica i dati dal file dei risultati di valutazione
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    successful_jailbreaks = []

    # Support both dict-of-lists (HarmBench) and flat list (StrongREJECT) formats
    if isinstance(data, dict):
        entries = [entry for results in data.values() for entry in results]
    else:
        entries = data

    for entry in entries:
        # Filtra solo quelli dove asr_e_label >= 0.5 (float) o == 1 (int)
        asr_e = entry.get('asr_e_label', 0)
        if asr_e == 1 or (isinstance(asr_e, float) and asr_e >= 0.5):
            # Logica di fallback: usa 'generation', se manca usa 'response'
            response_text = entry.get('generation') or entry.get('response', '')

            record = {
                'Behavior': entry.get('Behavior') or entry.get('vanilla prompt', ''),
                'Refined_behavior': entry.get('Refined_behavior') or entry.get('refine prompt', ''),
                'Response': response_text
            }
            successful_jailbreaks.append(record)

    # --- Salvataggio in JSON ---
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
    process_jailbreaks(args.input, args.output_json, args.output_csv)