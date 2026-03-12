from collections import Counter

from itzi.array_definitions import ARRAY_DEFINITIONS, ArrayCategory


def test_array_definitions():
    # Empty names is acceptable in internal arrays
    for attr in ["csdms_name", "cf_name"]:
        all_values = [
            getattr(arr_def, attr)
            for arr_def in ARRAY_DEFINITIONS
            if ArrayCategory.INTERNAL not in arr_def.category
        ]
        print(all_values)
        # No empty name
        if "" in all_values and not attr == "cf_name":
            print(attr)
            assert False, f"Found empty names in <{attr}>."
        # Make sure there is no duplicates
        values_counts = Counter(all_values)
        duplicates = [item for item, count in values_counts.items() if count > 1]
        if 0 < len(duplicates):
            if attr == "cf_name" and len(duplicates) == 1 and "" in duplicates:
                continue
            assert False, f"Found duplicates in <{attr}>: {duplicates}"

    # All arrays must have a unique key and description
    for attr in ["key", "description"]:
        all_values = [getattr(arr_def, attr) for arr_def in ARRAY_DEFINITIONS]
        print(all_values)
        # No empty name
        if "" in all_values and not attr == "cf_name":
            print(attr)
            assert False, f"Found empty names in <{attr}>."
        # Make sure there is no duplicates
        values_counts = Counter(all_values)
        duplicates = [item for item, count in values_counts.items() if count > 1]
        if 0 < len(duplicates):
            if attr == "cf_name" and len(duplicates) == 1 and "" in duplicates:
                continue
            assert False, f"Found duplicates in <{attr}>: {duplicates}"
