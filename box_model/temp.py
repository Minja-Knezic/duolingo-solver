import os

# Map (filename pattern, old index) to new class name and new index (0-6) for second pass
index_map = {
    ("type_sentence_", 8): ("type_sentence_sentence", 0),
    ("select_meaning_", 9): ("select_meaning_word", 1),
    ("select_meaning_", 10): ("select_meaning_option", 2),
    ("fill_blank_", 11): ("fill_blank_sentence", 3),
    ("fill_blank_", 12): ("fill_blank_add", 4),
    ("choose_words_", 13): ("choose_word_question", 5),
    ("choose_words_", 14): ("choose_word_option", 6),
    # Add more if needed
}

dirs_to_scan = ["box_training/labels/train", "box_training/labels/val"]

for dir_path in dirs_to_scan:
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if fname.endswith(".txt"):
                # Find all (pattern, old_index) pairs that match this filename
                relevant_pairs = [(pattern, old_index) for (pattern, old_index) in index_map if fname.startswith(pattern)]
                if not relevant_pairs:
                    continue
                fpath = os.path.join(root, fname)
                with open(fpath, "r") as f:
                    lines = f.readlines()
                new_lines = []
                replaced_count = 0
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    replaced = False
                    if parts:
                        first_val = parts[0]
                        for pattern, old_index in relevant_pairs:
                            if int(first_val) == old_index:
                                new_index = index_map[(pattern, old_index)][1]
                                parts[0] = str(new_index)
                                new_lines.append(" ".join(parts) + "\n")
                                print(f"{fpath}: Line {i} replaced {old_index} -> {new_index}")
                                replaced_count += 1
                                replaced = True
                                break
                        if not replaced:
                            new_lines.append(line)
                            print(f"{fpath}: Line {i} skipped (first value: {first_val})")
                    else:
                        new_lines.append(line)
                        print(f"{fpath}: Line {i} skipped (empty line)")
                with open(fpath, "w") as f:
                    f.writelines(new_lines)
                print(f"{fpath}: Total replacements: {replaced_count}")

print("All annotation files updated with correct new indices.")