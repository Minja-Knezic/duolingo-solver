import os

# Map (filename pattern, old index) to new class name and new index
index_map = {
    ("type_sentence_", 15): ("type_sentence_sentence", 8),
    ("select_meaning_", 15): ("select_meaning_word", 9),
    ("select_meaning_", 16): ("select_meaning_option", 10),
    ("fill_blank_", 15): ("fill_blank_sentence", 11),
    ("fill_blank_", 16): ("fill_blank_add", 12),
    ("choose_words_", 15): ("choose_word_question", 13),
    ("choose_words_", 16): ("choose_word_option", 14),
    # Add more if needed
}

dirs_to_scan = ["box_training/labels/train", "box_training/labels/val"]

for dir_path in dirs_to_scan:
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if fname.endswith(".txt"):
                for pattern, old_index in index_map:
                    if fname.startswith(pattern):
                        fpath = os.path.join(root, fname)
                        with open(fpath, "r") as f:
                            lines = f.readlines()
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if parts and int(parts[0]) == old_index:
                                # Replace old index with new index
                                new_index = index_map[(pattern, old_index)][1]
                                parts[0] = str(new_index)
                                new_lines.append(" ".join(parts) + "\n")
                            else:
                                new_lines.append(line)
                        with open(fpath, "w") as f:
                            f.writelines(new_lines)
                        print(f"Updated {fpath} using pattern {pattern} and old index {old_index}")
                        break  # Only match one pattern per file

print("All annotation files updated with correct new indices.")