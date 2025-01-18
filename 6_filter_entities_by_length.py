import json


def filter_entities_by_length(input_file, output_file, min_length=9):
    """
    Filter entities in each sentence based on the minimum length.
    Args:
        input_file (str): Path to the input file containing JSON data.
        output_file (str): Path to the output file to save the filtered results.
        min_length (int): Minimum length for filtering entities.
    """
    filtered_results = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

        for entry in data:
            sentence = entry["sentence"]
            ner = entry["ner"]

            # Filter entities based on length
            filtered_entities = [entity for entity in ner if
                                 len("".join([sentence[idx] for idx in entity["index"]])) > min_length]

            # Add to results if there are any entities left after filtering
            if filtered_entities:
                filtered_results.append({
                        "sentence": sentence,
                        "ner": filtered_entities
                    })

    # Write filtered results to output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(filtered_results, outfile, ensure_ascii=False, indent=None)


# Define file paths
input_file = 'D:\PPSUC\workspace\BRCNN_KE\data\cdtier\\total.json'
output_file = 'D:\PPSUC\workspace\BRCNN_KE\data\cdtier\\total_len_10.json'

# Run the filtering function
filter_entities_by_length(input_file, output_file)
