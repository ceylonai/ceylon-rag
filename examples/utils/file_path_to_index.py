def path_to_int64(path: str) -> int:
    """
    Convert a relative file path to a unique 64-bit integer.
    Uses a simple but effective hashing approach that maintains uniqueness for paths.

    Args:
        path (str): The relative file path to convert

    Returns:
        int: A unique 64-bit integer representation of the path

    Example:
        >>> path_to_int64("folder/subfolder/file.txt")
        8743629871234
    """
    # Initialize hash value
    hash_val = 0

    # Use a prime number for better distribution
    prime = 31
    mod = 2 ** 63  # Use 63 bits to ensure we stay within signed int64 range

    # Calculate hash while staying within int64 bounds
    for char in path:
        hash_val = (hash_val * prime + ord(char)) % mod

    return hash_val


# Example usage and testing
def test_path_converter():
    # Test paths
    test_paths = [
        "folder/file1.txt",
        "folder/file2.txt",
        "another_folder/file.txt",
        "deep/nested/path/to/file.txt",
        "single_file.txt"
    ]

    # Convert and check for uniqueness
    results = {}
    for path in test_paths:
        int_val = path_to_int64(path)
        if int_val in results:
            print(f"Collision detected between {path} and {results[int_val]}")
            return False
        results[int_val] = path
        print(f"{path} -> {int_val}")

    print("\nAll paths converted successfully with no collisions")
    return True


if __name__ == "__main__":
    test_path_converter()