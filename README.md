# Workplace Converter

## Description

This code was written for the analysis and conversion of JSON datasets related to workplace data for a client. The primary purpose is to read JSON files, extract relevant data, and convert it into structured formats like CSV and PKL for further analysis.

## Features

- **Read JSON Files**: Recursively read JSON files from a specified directory.
- **Extract Data**: Extract and process data from JSON files into structured records.
- **Convert to DataFrame**: Convert extracted records into a Pandas DataFrame with MultiIndex columns.
- **Save as PKL and CSV**: Save the DataFrame as both PKL and CSV files.

## Project Structure

```
workplace_converter/
├── meta_workplace_converter.py
├── README.md
├── poetry.lock
└── pyproject.toml
```

## Key Functions

### `read_json_files(root_dir)`

Reads JSON files from the specified directory and returns a dictionary of the data.

**Parameters:**
- `root_dir`: Path to the directory containing JSON files.

**Returns:** 
- Dictionary containing the JSON data.


### `extract_data(data, file_name="")`

Extracts and processes data from the given JSON data into structured records.

**Parameters:**
- `data`: The JSON data to be processed.
- `file_name`: Optional file name for reference.

**Returns:** 
- Pandas DataFrame with MultiIndex columns.

### `convert_to_pkl(root_dir)`

Converts JSON files in the specified directory to PKL and CSV formats.

**Parameters:**
- `root_dir`: Path to the directory containing JSON files.

**Returns:** 
- None. Saves the converted files to the output directory.

## Notes

- The code handles nested structures within JSON files, including vectors and dictionaries.
- The output directory structure mirrors the input directory structure, with a suffix `_pkl_converted`.
- The code ensures that redundant records are not appended to the final DataFrame.

## How to use
- Make sure you have [Poetry](https://python-poetry.org/) installed, then navigate to the project directory and run:
  ```
  poetry install
  ```
- To convert your workplace data, put your folder containing the workplace JSON files in the project root directory, rename the `root_dir` in the second last line of [meta_workplace_converter.py](/meta_workplace_converter.py) to your folder name, and run 
  ```
  python meta_workplace_converter.py
  ```
  The extracted files will be saved to a new folder named `<your_workplace_data_folder_name>_pkl_converted` in both `.pkl` and `.csv` formats.