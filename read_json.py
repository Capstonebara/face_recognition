import json
import numpy as np
import os


# Load pre-computed embeddings from JSON files
def load_precomputed_embeddings(json_folder_path: str):
    """
    Load precomputed embeddings stored in JSON format.

    :param json_folder_path: Path to folder containing the JSON files with embeddings.
    :return: A tuple containing embeddings (numpy array) and labels (list of usernames or image names).
    """
    embeddings = []
    labels = []

    # Iterate over all JSON files in the directory
    for filename in os.listdir(json_folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(json_folder_path, filename)
            print(f"Loading file: {file_path}")  # Debug: Print the file being loaded
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    print(
                        f"Loaded data from {filename}"
                    )  # Debug: Print confirmation of file load
                except json.JSONDecodeError as e:
                    print(f"Error loading {filename}: {e}")
                    continue  # Skip this file if it couldn't be loaded

                # Debugging: Check structure and print a few samples
                print(f"Contents of {filename}:")
                for image_name, embedding_data in list(data.items())[
                    :2
                ]:  # Print first two items
                    print(
                        f"  Image: {image_name}, Embedding: {embedding_data['vector'][:5]}..."
                    )  # Show first 5 values of the vector

                # Print how many embeddings are in this JSON file
                num_embeddings = len(data)
                print(f"Number of embeddings in {filename}: {num_embeddings}")

                # Process the embeddings
                for image_name, embedding_data in data.items():
                    embeddings.append(
                        embedding_data["vector"]
                    )  # The embedding vector for each image
                    labels.append(
                        image_name
                    )  # Use the image name or username as the label

    embeddings_np = np.array(embeddings).astype("float32")  # Convert to numpy array
    print(
        f"Total embeddings loaded: {len(embeddings_np)}"
    )  # Debug: Print total embeddings
    print(
        f"Sample embeddings: {embeddings_np[:2]}"
    )  # Debug: Print first two embeddings
    return embeddings_np, labels


# Test the loading function
if __name__ == "__main__":
    json_folder_path = "/home/vanellope/face_recognition_project/embedding/"  # Path where your JSON files are stored
    embeddings, labels = load_precomputed_embeddings(json_folder_path)

    # Optionally, print out a few details about the loaded embeddings
    print("Loaded embeddings shape:", embeddings.shape)
    print("Loaded labels:", labels[:10])  # Print first 10 labels to check
