import numpy as np


def matrix_multiplication(matrix1, matrix2):
    return np.matmul(matrix1, matrix2)


def multiplication_check(matrices):
    starting_matrix = matrices[0]
    if np.isscalar(starting_matrix[0]):
        prev_matrix_columns = 1
    else:
        prev_matrix_columns = len(starting_matrix[0])
    for i in range(1, len(matrices)):
        current_matrix = matrices[i]
        curr_matrix_rows = len(current_matrix)
        if prev_matrix_columns != curr_matrix_rows:
            return False
        if np.isscalar(starting_matrix[0]):
            prev_matrix_columns = 1
        else:
            prev_matrix_columns = len(starting_matrix[0])
    return True


def multiply_matrices(matrices):
    try:
        return np.linalg.multi_dot(matrices)
    except ValueError:
        return None


def compute_multidimensional_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)


def compute_2d_distance(arr1, arr2):
    return compute_multidimensional_distance(arr1, arr2)


def compute_pair_distances(matrix):
    x2 = np.sum(matrix ** 2, axis=1)
    y2 = np.sum(matrix ** 2, axis=1)
    xy = np.matmul(matrix, matrix.T)
    x2 = x2.reshape(-1, 1)
    return np.sqrt(x2 - 2 * xy + y2)


if __name__ == "__main__":
    array1 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    array2 = np.diag([1, 2, 3, 4, 5])
    array3 = np.ones((5, 5))




