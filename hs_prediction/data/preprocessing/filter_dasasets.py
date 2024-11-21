"""filter datasets
"""

import os
import hydra


def filter_elements(elements_path, invalid_path, output_path):
    """Filter out invalid elements

    Args:
        elements_path: path to the protein elements
        invalid_path: path to list of invalid elements to filter out
        output_path: path to save the filtered elements
    """
    with open(elements_path, "r") as train_file:
        elements = train_file.read().splitlines()

    with open(invalid_path, "r") as invalid_file:
        invalid_elements = set(invalid_file.read().splitlines())

    filtered_elements = [
        element for element in elements if element not in invalid_elements
    ]
    with open(output_path, "w") as filtered_file:
        filtered_file.write("\n".join(filtered_elements))


@hydra.main(
    config_path="../../../config", config_name="location_model", version_base="1.1"
)
def main(config):
    dataset_parent_path = os.path.dirname(config.data.dataset_path)
    train_path = os.path.join(dataset_parent_path, "splits/train.txt")
    test_path = os.path.join(dataset_parent_path, "splits/test.txt")
    invalid_path = os.path.join(dataset_parent_path, "splits/problematic_proteins.txt")
    output_path = os.path.join(dataset_parent_path, "splits/train_filtered.txt")
    filter_elements(train_path, invalid_path, output_path)
    output_path = os.path.join(dataset_parent_path, "splits/test_filtered.txt")
    filter_elements(test_path, invalid_path, output_path)


if __name__ == "__main__":
    main()
