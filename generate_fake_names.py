import csv
import string
from nltk.corpus import names
import nltk
import random

# Download the names corpus if you haven't already
nltk.download('names')

def generate_fake_names():
    # Get all first names from NLTK corpus
    male_names = names.words('male.txt')
    female_names = names.words('female.txt')
    all_names = set(male_names + female_names)
    
    # Get all 26 uppercase letters for initials
    letters = list(string.ascii_uppercase)
    
    # Create a list to store all combinations
    fake_names = []
    
    # Generate all combinations of names and initials
    for first_name in all_names:
        for initial in letters:
            fake_names.append((first_name, initial))
    
    return fake_names

def write_to_csv(filename, fake_names):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Name', 'Initial', 'Full_Name'])  # Header
        
        for first_name, initial in fake_names:
            full_name = f"{first_name} {initial}"
            writer.writerow([first_name, initial, full_name])

if __name__ == "__main__":
    # Generate all fake names
    print("Generating fake names...")
    fake_names = generate_fake_names()
    
    fake_names.sort(key=lambda x: len(x[0]))  # Sort by length of first name
    
    # Write to CSV file
    csv_filename = "output/fake_names.csv"
    print(f"Writing {len(fake_names)} fake names to {csv_filename}...")
    write_to_csv(csv_filename, fake_names)
    
    print("Done!")