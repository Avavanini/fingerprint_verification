"""
generate_dataset_info.py — Parse SOCOFing filenames and generate dataset_info.csv

SOCOFing filename format: {subject_id}__{gender}_{hand}_{finger}_finger.BMP
Example: 100__M_Left_index_finger.BMP
"""

import os
import csv
import re
from pathlib import Path


def parse_filename(filename: str) -> dict:
    """Parse a SOCOFing filename into its components."""
    # Pattern: {id}__{gender}_{hand}_{finger}_finger.BMP
    match = re.match(r'^(\d+)__([MF])_(\w+)_(\w+)_finger\.BMP$', filename)
    if not match:
        return None
    
    subject_id = int(match.group(1))
    gender = 'Male' if match.group(2) == 'M' else 'Female'
    hand = match.group(3)
    finger = match.group(4)
    finger_index = f"{hand}_{finger}"
    
    return {
        'subject_id': subject_id,
        'gender': gender,
        'hand': hand,
        'finger': finger,
        'finger_index': finger_index,
    }


def generate_csv(data_dir: str, output_path: str):
    """Generate dataset_info.csv from SOCOFing directory."""
    rows = []
    
    # Process Real images
    real_dir = os.path.join(data_dir, 'SOCOFing', 'Real')
    if os.path.exists(real_dir):
        for filename in sorted(os.listdir(real_dir)):
            if not filename.endswith('.BMP'):
                continue
            info = parse_filename(filename)
            if info:
                info['image_path'] = os.path.join('data', 'raw', 'SOCOFing', 'Real', filename)
                info['category'] = 'Real'
                info['alteration'] = 'none'
                rows.append(info)
    
    # Process Altered images
    altered_dir = os.path.join(data_dir, 'SOCOFing', 'Altered')
    if os.path.exists(altered_dir):
        for difficulty in ['Altered-Easy', 'Altered-Medium', 'Altered-Hard']:
            alt_subdir = os.path.join(altered_dir, difficulty)
            if not os.path.exists(alt_subdir):
                continue
            for filename in sorted(os.listdir(alt_subdir)):
                if not filename.endswith('.BMP'):
                    continue
                # Altered filenames may have extra info, try basic parse
                # Pattern: {id}__{gender}_{hand}_{finger}_finger_{alteration}.BMP
                base_match = re.match(r'^(\d+)__([MF])_(\w+?)_(\w+?)_finger', filename)
                if base_match:
                    subject_id = int(base_match.group(1))
                    gender = 'Male' if base_match.group(2) == 'M' else 'Female'
                    hand = base_match.group(3)
                    finger = base_match.group(4)
                    rows.append({
                        'subject_id': subject_id,
                        'gender': gender,
                        'hand': hand,
                        'finger': finger,
                        'finger_index': f"{hand}_{finger}",
                        'image_path': os.path.join('data', 'raw', 'SOCOFing', 'Altered', difficulty, filename),
                        'category': 'Altered',
                        'alteration': difficulty.replace('Altered-', '').lower(),
                    })
    
    # Write CSV
    fieldnames = ['image_path', 'subject_id', 'gender', 'hand', 'finger', 'finger_index', 'category', 'alteration']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Generated {output_path} with {len(rows)} entries")
    
    # Summary stats
    real_count = sum(1 for r in rows if r['category'] == 'Real')
    altered_count = sum(1 for r in rows if r['category'] == 'Altered')
    subjects = set(r['subject_id'] for r in rows)
    print(f"  Real images: {real_count}")
    print(f"  Altered images: {altered_count}")
    print(f"  Total subjects: {len(subjects)}")
    print(f"  Subject ID range: {min(subjects)} - {max(subjects)}")


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dataset_info.csv')
    generate_csv(data_dir, output_path)
