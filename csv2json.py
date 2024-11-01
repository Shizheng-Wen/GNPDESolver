import csv
import json
import ast
import argparse
import os

def csv_to_json(csv_filename, json_filename, row_number=0):
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = list(csv.DictReader(csvfile))
        if row_number >= len(reader):
            print(f"Row number {row_number} does not exist in file {csv_filename}, skipping this file.")
            return
        row = reader[row_number]
        json_data = {}
        # Process the specified fields
        for key in ['setup', 'graph', 'model', 'dataset', 'optimizer', 'path']:
            value = row.get(key, '')
            if value.strip() == '':
                continue  # Skip empty values
            if value.startswith("{") and value.endswith("}"):
                # Parse the dictionary string using ast.literal_eval
                parsed_value = ast.literal_eval(value)
                if key == 'path':
                    # Remove the unwanted prefix from all path values
                    prefix = '/cluster/scratch/shiwen/GNPDESolver/'
                    parsed_value = {k: v.replace(prefix, '') for k, v in parsed_value.items()}
                json_data[key] = parsed_value
            else:
                # Keep the value as a string
                json_data[key] = value

        # Custom JSON encoder to format lists on a single line
        class CompactJSONEncoder(json.JSONEncoder):
            def __init__(self, *args, **kwargs):
                super(CompactJSONEncoder, self).__init__(*args, **kwargs)
                self.indent = kwargs.get('indent', None)
                self.sort_keys = kwargs.get('sort_keys', False)

            def encode(self, o):
                def _encode(o, level):
                    if isinstance(o, dict):
                        items = []
                        for k, v in (sorted(o.items()) if self.sort_keys else o.items()):
                            items.append('\n' + ' ' * (level * self.indent) + json.dumps(k) + ': ' + _encode(v, level + 1))
                        s = '{' + ','.join(items) + '\n' + ' ' * ((level - 1) * self.indent) + '}'
                        return s
                    elif isinstance(o, list):
                        # Lists are output on a single line
                        s = '[' + ', '.join(_encode(e, level + 1) for e in o) + ']'
                        return s
                    else:
                        return json.dumps(o)

                return _encode(o, 1)
        
        # Write the data to a JSON file with indentation
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=4, ensure_ascii=False, cls=CompactJSONEncoder)
        print(f"Processed file {csv_filename} row {row_number}, output saved to {json_filename}")

def process_folder(csv_folder, output_folder, row_number=0):
    csv_folder = os.path.normpath(csv_folder)
    prefix_to_remove = os.path.normpath(csv_folder).split(os.sep)[0]
    for root, dirs, files in os.walk(csv_folder):
        for filename in files:
            if filename.endswith('.csv'):
                csv_filepath = os.path.normpath(os.path.join(root, filename))
                # Remove the first component from csv_filepath
                relative_parts = csv_filepath.split(os.sep)
                if relative_parts[0] == prefix_to_remove:
                    relative_parts = relative_parts[1:]  # Remove the first component
                relative_path = os.path.join(*relative_parts)
                # Now construct the output path
                json_output_folder = os.path.join(output_folder, os.path.dirname(relative_path))
                if not os.path.exists(json_output_folder):
                    os.makedirs(json_output_folder)
                json_filename = os.path.splitext(filename)[0] + '.json'
                json_filepath = os.path.join(json_output_folder, json_filename)
                csv_to_json(csv_filepath, json_filepath, row_number)

def main():
    parser = argparse.ArgumentParser(description='Convert CSV files to JSON format')
    parser.add_argument('-c', '--csv_file', help='Path to the CSV file to process')
    parser.add_argument('-f', '--csv_folder', help='Path to the folder containing CSV files to process')
    parser.add_argument('-o', '--output_folder', default='config', help='Output folder for JSON files')
    parser.add_argument('-r', '--row', type=int, default=0, help='Row number to process in the CSV file (starting from 0)')
    args = parser.parse_args()

    if args.csv_file:
        # Process a single file
        csv_filename = args.csv_file
        # Construct the output JSON filename
        relative_path = os.path.relpath(os.path.dirname(csv_filename), os.path.dirname(args.csv_file))
        json_output_folder = os.path.join(args.output_folder, relative_path)
        if not os.path.exists(json_output_folder):
            os.makedirs(json_output_folder)
        json_filename = os.path.splitext(os.path.basename(csv_filename))[0] + '.json'
        json_filepath = os.path.join(json_output_folder, json_filename)
        csv_to_json(csv_filename, json_filepath, args.row)
    elif args.csv_folder:
        # Process all CSV files in the folder
        csv_folder = args.csv_folder
        output_folder = args.output_folder
        process_folder(csv_folder, output_folder, args.row)
    else:
        print("Please specify a CSV file with -c or a CSV folder with -f")
        parser.print_help()

if __name__ == '__main__':
    main()