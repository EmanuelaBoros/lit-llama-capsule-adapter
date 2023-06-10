import os
import bz2
import json
import jsonlines
from multiprocessing import Pool
from transformers import AutoTokenizer

enc = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-6.9b-deduped",
)

def get_token_count(text):
    return len(enc.tokenize(text))
# Get list of files
# Get list of files for each subfolder
def get_files(folder):
    subfolder_file_lists = {}
    for subdir, dirs, files in os.walk(folder):
        if subdir == folder:  # Skip the root folder
            continue
        file_list = []
        for file in files:
            if file.endswith(".jsonl.bz2"):
                # construct full file path
                file_path = subdir + os.sep + file
                file_list.append(file_path)
        subfolder_file_lists[subdir] = file_list
    return subfolder_file_lists

# Process each file
def process_data(file_list):
    documents = []
    for file_path in file_list:
        print(file_path)
        with bz2.open(file_path, "rt") as bz_file:
            # jsonlines reader
            reader = jsonlines.Reader(bz_file)
            document = ""
            for obj in reader:
                # check if 'ft' is in the json object
                if 'ft' in obj:
                    # concatenate the 'ft' string to the current document
                    document += obj['ft']
                    print(obj['ft'])
            # append the current document to the list of documents
            documents.append(document)
    return documents

# Call the functions
subfolder_file_lists = get_files('data')

token_counts = {}
# token_count = 0
# subfolder_documents = {}
for subdir, file_list in subfolder_file_lists.items():
    documents = []
    print(subdir, ':', len(file_list), 'documents')
    from tqdm import tqdm
    with open(os.path.join('data', subdir.split('/')[-1] + '.txt'), 'w') as f:
        for file_path in tqdm(file_list, total=len(file_list)):
            with bz2.open(file_path, "rt") as bz_file:
                # jsonlines reader
                reader = jsonlines.Reader(bz_file)
                document = ""
                for obj in reader:
                    # check if 'ft' is in the json object
                    if 'ft' in obj:
                        # concatenate the 'ft' string to the current document
                        document += obj['ft']
                # append the current document to the list of documents
                documents.append(document)
        f.write('\n\n'.join(documents))
    # subfolder_documents[subdir] = documents
    token_count = get_token_count(' '.join(documents))
    token_counts[subdir] = token_count
    print(f"[INFO] Got {token_count} tokens for {subdir}.")

    summ = 0
    for i in token_counts:
        print(f"{i}: {token_counts[i]}")
        summ += token_counts[i]
    print("Total: ", summ)

with open(os.path.join('data', 'token_counts_impresso.json'), "w") as f:
    json.dump(token_counts, f)
