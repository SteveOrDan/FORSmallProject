import sys
import os
import csv
import shutil
from collections import Counter

FILES_TO_PROCESS = ["Building1.csv", "Building2.csv", "Building4.csv"]
CLEAN_DIR = "Clean"


def prepare_clean_directory():
    if os.path.exists(CLEAN_DIR):
        # directory exists -> clear contents
        for filename in os.listdir(CLEAN_DIR):
            file_path = os.path.join(CLEAN_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)       # remove file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)   # remove folder
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    else:
        os.makedirs(CLEAN_DIR)  # create directory if it doesn't exist


def process_file(path: str) -> None:
    if not os.path.isfile(path):
        print(f"[WARN] {path} does not exist or is not a file, skipping.")
        return

    # output path inside Clean/
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(CLEAN_DIR, f"{base}_dedup.csv")

    # Read file
    with open(path, newline="") as f:
        lines = f.readlines()

    # Separate comment lines and data lines
    comments = []
    data_lines = []
    header_found = False

    for line in lines:
        if not header_found and line.lstrip().startswith("#"):
            comments.append(line.rstrip("\n"))
        else:
            header_found = True
            data_lines.append(line)

    reader = csv.reader(data_lines)
    try:
        _ = next(reader)  # skip old header
    except StopIteration:
        print(f"[INFO] {path}: no data rows, skipping.")
        return

    seen = set()
    unique_rows = []
    removed = []

    for row in reader:
        if not row or all(cell.strip() == "" for cell in row):
            continue
        try:
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])
        except (ValueError, IndexError):
            continue

        key = (x, y, z)
        if key in seen:
            removed.append(key)
        else:
            seen.add(key)
            unique_rows.append((x, y, z))

    # Write new clean CSV
    with open(out_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["x", "y", "z"])  # assignment-required format
        for x, y, z in unique_rows:
            writer.writerow([x, y, z])

    # Print summary
    print(f"{os.path.basename(path)} → {out_path}")

    if removed:
        cnt = Counter(removed)
        print(f"  Removed {len(removed)} duplicate row(s).")
        print("  Duplicated coordinates (coord: times removed):")
        for coord, c in cnt.items():
            print(f"    {coord}: {c}")
    else:
        print("  No duplicates found.")


def main():
    prepare_clean_directory()

    for path in FILES_TO_PROCESS:
        process_file(path)


if __name__ == "__main__":
    main()
