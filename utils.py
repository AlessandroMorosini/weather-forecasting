# utils.py

def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))