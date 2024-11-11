# workplace_converter

## Features
- Convert JSON files to tabular format.
- Search for specific strings within JSON files.
- Convert JSON files to pickle format for efficient storage and retrieval.

## Project Structure
```
workplace_converter/
├── dev_notebook.ipynb
├── README.md
```

## Key Functions

### `search_strings_in_json(directory, strings_to_find)`
Search for specific strings within JSON files in a directory.
- **Parameters**:
  - `directory`: Path to the directory containing JSON files.
  - `strings_to_find`: List of strings to search for.
- **Returns**: List of tuples with file paths and matching strings.

### `read_json_files(root_dir)`
Read JSON files from a directory.
- **Parameters**:
  - `root_dir`: Path to the directory or JSON file.
- **Returns**: Dictionary with file names and their JSON content.

### `extract_data(data)`
Extract data from JSON content into a tabular format.
- **Parameters**:
  - `data`: Dictionary with JSON content.
- **Returns**: Pandas DataFrame with extracted data.

### `convert_to_csv(root_dir)`
Convert JSON files in a directory to pickle format.
- **Parameters**:
  - `root_dir`: Path to the directory containing JSON files.

## Notes
- Ensure all JSON files are UTF-8 encoded.
- Handle errors gracefully when reading malformed JSON files.
- Use appropriate directories for different types of JSON data.
