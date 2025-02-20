import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from copy import deepcopy
import os
import json
import pandas as pd
import pickle


def read_json_files(root_dir):
    data = {}
    if root_dir.endswith('.json'):
        temp_key_name = root_dir.split('/')[-1].split('.')[0]
        with open(root_dir, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
                data[temp_key_name] = json_data
            except json.JSONDecodeError as e:
                print(f"Error reading {root_dir}: {e}")
    else:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.json'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        temp_key_name = root_dir + '/' + file
                        try:
                            json_data = json.load(f)
                            data[temp_key_name] = json_data
                        except json.JSONDecodeError as e:
                            print(f"Error reading {filepath}: {e}")
    return data

def extract_data(data, file_name = ""):
    records = []
    columns = set()

    def check_exists_append(record, key, value, columns, force_new=False, is_title=False):
        """
        Append function that combines values into lists unless force_new is True
        For titles, always overwrite instead of appending
        """
        if is_title:
            record[key] = value  # Simply overwrite for titles
            columns.add(key)
        elif force_new:
            record[key] = value
            columns.add(key)
        else:
            if record.get(key, False):
                if isinstance(record[key], list):
                    record[key].append(value)
                else:
                    record[key] = [record[key]] + [value]
            else:
                record[key] = value
                columns.add(key)

    def process_vec_items(vec_items, base_record, key_name, parent_labels):
        """Helper function to process vector items recursively"""
        if vec_items:
            for nested_item in vec_items:
                process_data({'label_values': [nested_item]}, base_record.copy(), key_name)

    def has_nested_dict(dict_items):
        """Check if there are nested dictionaries in the dict_items"""
        for entry in dict_items:
            if 'dict' in entry:
                return True
        return False

    def process_dict_items(dict_items, base_record, ent_field_name, labels, parent_title=None):
        """Helper function to process dictionary items recursively"""
        has_nested = has_nested_dict(dict_items)
        current_record = base_record.copy()
        
        # Add parent title to record if it exists
        if parent_title:
            title_key = tuple([ent_field_name, 'title'])
            check_exists_append(current_record, title_key, parent_title, columns, is_title=True)
        
        for entry in dict_items:
            value = entry.get('value') or entry.get('timestamp_value')
            temp_label = entry.get('label', '')
            nested_ent_field_name = entry.get('ent_field_name', '')
            
            # If there's an ent_field_name in the entry, use it
            current_field_name = nested_ent_field_name if nested_ent_field_name else ent_field_name
            
            # Handle nested vectors within dictionary
            if 'vec' in entry:
                nested_key_name = labels + [temp_label or current_field_name]
                process_vec_items(entry['vec'], current_record, nested_key_name, labels)
            # Handle nested dictionaries
            elif 'dict' in entry:
                nested_labels = labels + [temp_label or current_field_name]
                process_dict_items(entry['dict'], current_record, current_field_name, 
                                nested_labels, parent_title)
            # Handle leaf values
            else:
                if labels:
                    key = tuple(labels + [temp_label or current_field_name])
                else:
                    key = tuple([current_field_name, temp_label]) if temp_label else tuple([current_field_name, current_field_name])
                
                check_exists_append(current_record, key, value, columns, force_new=has_nested)
        
        # Only append if record has data
        if len(current_record) > 0:
            records.append(current_record)

    def process_data(item, parent_record, parent_labels):
        record = parent_record.copy()
        labels = parent_labels.copy()

        # Extract basic fields
        for key in ['timestamp', 'media', 'fbid', 'ent_name']:
            if key in item:
                record[key] = item[key]
                columns.add((key, ""))

        # Process 'label_values'
        label_values = item.get('label_values', [])
        if label_values:
            for lv in label_values:
                ent_field_name = lv.get('ent_field_name', '')
                label = lv.get('label', '')
                key_name = [ent_field_name, label] if label else [ent_field_name, ent_field_name]
                
                if 'value' in lv or 'timestamp_value' in lv:
                    value = lv.get('value') or lv.get('timestamp_value')
                    if labels:
                        check_exists_append(record, tuple(labels), value, columns)
                    else:
                        check_exists_append(record, tuple(key_name), value, columns)
                elif 'vec' in lv:
                    # Handle vector items recursively
                    process_vec_items(lv['vec'], record, key_name, labels)
                elif 'dict' in lv:
                    # Extract title before processing dict
                    dict_title = lv.get('title')
                    # Handle dictionary items recursively with title
                    process_dict_items(lv['dict'], record, ent_field_name, 
                                    labels if labels else key_name,
                                    dict_title)
                    # If dict is empty but has title, create a record with just the title
                    if not lv['dict'] and dict_title:
                        title_record = record.copy()
                        title_key = tuple([ent_field_name, 'title'])
                        check_exists_append(title_record, title_key, dict_title, columns, is_title=True)
                        records.append(title_record)
                    continue  # Skip the final record append as it's handled in process_dict_items

        else:
            # Handle case for generic JSON file
            for key, value in item.items():
                check_exists_append(record, tuple([key]), value, columns)

        # Only append if record has data and is not already included
        if len(record) > 0 and record not in records:
            records.append(record.copy())

    # Iterate through data
    if isinstance(data, dict):
        for data_value in data.values():
            if isinstance(data_value, list):
                for item in data_value:
                    process_data(item, {}, [])
            else:
                process_data(data_value, {}, [])

    def create_multiindex_df(data):
        if not data:
            # Return empty DataFrame with proper structure
            return pd.DataFrame(columns=pd.MultiIndex.from_tuples([('empty', '')]))
            
        # Function to flatten dictionary and preserve both tuple elements
        def flatten_dict(d):
            flattened = {}
            for k, v in d.items():
                if isinstance(k, tuple):
                    # Store both parts of the tuple
                    flattened[k] = v
                else:
                    # For non-tuple keys, create a tuple with same value
                    flattened[(k, k)] = v
            return flattened
        
        # Flatten all dictionaries
        flattened_data = [flatten_dict(d) for d in data]
        
        # Create initial DataFrame
        df = pd.DataFrame(flattened_data)
        
        if df.empty:
            return df
            
        # Get all unique column tuples
        columns = df.columns.tolist()
        
        # Create MultiIndex columns
        multi_index = pd.MultiIndex.from_tuples(columns)
        
        # Create final DataFrame with MultiIndex
        final_df = pd.DataFrame(df.values, columns=multi_index)
        
        return final_df
    
    return create_multiindex_df(records)


def convert_to_pkl(root_dir):
    assert not root_dir.endswith('.json'), f"ERROR: Path must be a directory not a file {root_dir}"
    root_temp = ""
    folder_loc = -1
    new_name = "default_csv"
    
    for root, dirs, files in os.walk(root_dir):
        # On first iteration we're setting up temp variables
        if not root_temp:
            root_temp = root.split("/")
            folder_loc = len(root_temp) - 1
            new_name = root_temp[folder_loc] + "_pkl_converted"
        
        # For each file in each directory
        for file in files:
            if file.endswith('.json'):
                # Create new output path
                output_root = root.split("/")
                output_root[folder_loc] = new_name
                output_root = "/".join(output_root)
                
                # Process the file
                filepath = os.path.join(root, file)
                print(f"Processing {filepath}")
                
                temp_data = read_json_files(filepath)
                temp_records = extract_data(temp_data, file)
                
                # Create output directory if it doesn't exist
                os.makedirs(output_root, exist_ok=True)
                
                # Save each DataFrame in the dictionary
                output_filename = file.replace('.json', '.pkl')
                output_csv_filename = file.replace('.json', '.csv')
                output_path = os.path.join(output_root, output_filename)
                output_csv_path = os.path.join(output_root, output_csv_filename)
                temp_records.to_pickle(output_path, protocol=pickle.HIGHEST_PROTOCOL)
                temp_records.to_csv(output_csv_path)
                print(f"Saved {output_path}")

if __name__ == "__main__":
    root_dir = 'Workplace Data Company Information'
    convert_to_pkl(root_dir)