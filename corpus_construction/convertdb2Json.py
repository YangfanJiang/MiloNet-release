import json

def transform_json(filename, output_filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = {}
    for item in data:
        group_id = item["id"]
        if group_id not in groups:
            groups[group_id] = {}
        groups[group_id][item["key"]] = item["string_value"].strip()

    result = {}
    for group in groups.values():
        response = group.get("response")
        document = group.get("chroma:document")
        if response and document:
            result[response] = document

    with open(output_filename, "w", encoding="utf-8") as out_file:
        json.dump(result, out_file, ensure_ascii=False, indent=4)

mapping = transform_json("embedding_metadata.json", "doc_summary_dict.json")
