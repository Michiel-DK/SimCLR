import numpy as np

def check_duplicates(array1, array2):
    """
    Check if there are duplicate elements between two numpy arrays.

    Parameters:
        array1 (numpy.ndarray): First array.
        array2 (numpy.ndarray): Second array.

    Returns:
        bool: True if there are duplicates, False otherwise.
    """
    
    set1 = set(array1.flatten())
    set2 = set(array2.flatten())
    
    # Check for duplicates
    duplicates = set1.intersection(set2)
    if duplicates:
        print(f"Duplicate elements: {duplicates}")
        return True
    else:
        print("No duplicates found.")
        return False