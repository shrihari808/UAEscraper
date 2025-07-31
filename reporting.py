# reporting.py

import os
import json
import pandas as pd
import config

def flatten_json(y):
    """
    Flattens a nested JSON object into a single level.
    - Dictionaries are recursively flattened with keys concatenated by underscores.
    - Lists of objects (like PeopleIntelligence) are serialized into a JSON string
      to fit neatly into a single Excel cell.
    """
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            # Serialize lists into a JSON string to keep them in one cell
            out[name[:-1]] = json.dumps(x)
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def generate_excel_report():
    """
    Reads all individual JSON analysis files from the output directory,
    compiles them into a pandas DataFrame, and saves the result as an Excel file.
    """
    print("\n--- Generating Final Excel Report ---")
    
    json_dir = config.ANALYSIS_OUTPUT_DIR
    if not os.path.exists(json_dir):
        print(f"❌ Analysis directory '{json_dir}' not found. Please run the --analyze or --analyze-all step first.")
        return

    all_records = []
    json_files = [f for f in os.listdir(json_dir) if f.endswith('_analysis.json')]

    if not json_files:
        print("❌ No analysis JSON files found to process in the 'analysisJsons' directory.")
        return

    print(f"-> Found {len(json_files)} analysis files to compile into a report.")
    for file_name in json_files:
        # Extract the company name from the filename (e.g., "Company_Name_analysis.json")
        company_name_from_file = file_name.replace('_analysis.json', '').replace('_', ' ')
        file_path = os.path.join(json_dir, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if "error" in data:
                    print(f"⚠️  Skipping file {file_name} due to an analysis error inside.")
                    continue
                
                flat_record = flatten_json(data)
                flat_record['CompanyName'] = company_name_from_file  # Add the company name as a new column
                all_records.append(flat_record)
            except json.JSONDecodeError:
                print(f"⚠️  Could not parse JSON from {file_name}. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred while processing {file_name}: {e}")

    if not all_records:
        print("❌ No valid records could be processed from the JSON files.")
        return

    # Create DataFrame from the list of flattened records
    df = pd.DataFrame(all_records)
    
    # Reorder columns to ensure 'CompanyName' is the first column for easy identification
    if 'CompanyName' in df.columns:
        company_col = df.pop('CompanyName')
        df.insert(0, 'CompanyName', company_col)

    output_path = 'analysis_summary.xlsx'
    try:
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"\n✅ Success! Final report saved to '{output_path}'")
    except Exception as e:
        print(f"❌ Failed to save Excel file: {e}")
        print("   Please ensure you have 'openpyxl' installed (`pip install openpyxl`).")

if __name__ == '__main__':
    # This allows the script to be run directly if needed
    generate_excel_report()
