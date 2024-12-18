import numpy as np
from scipy.signal import convolve2d

def generate_random_matrix(size):
    return np.random.randint(0, 256, size, dtype=int)

def generate_random_kernel(kernel_size):
    return np.random.randint(-255, 256, kernel_size, dtype=int)

def apply_convolution(matrix, kernel):
    return convolve2d(matrix, kernel, mode='same', boundary='wrap')

def save_to_txt(matrix, kernel, output_matrix, filename):
    with open(filename, 'w') as f:
        np.savetxt(f, matrix, fmt='%d', delimiter=' ')
        np.savetxt(f, kernel, fmt='%d', delimiter=' ')
        np.savetxt(f, output_matrix, fmt='%d', delimiter=' ')

def main():
    sizes = [(10, 10), (50, 50), (100, 100), (200, 200)]
    kernel_sizes = [(3, 3), (5, 5), (10, 10)]
    file_counter = 1

    for size in sizes:
        matrix = generate_random_matrix(size)
        for kernel_size in kernel_sizes:
            kernel = generate_random_kernel(kernel_size)
            convolved_matrix = apply_convolution(matrix, kernel)

            filename = f"convolution_result_{file_counter}.txt"
            save_to_txt(matrix, kernel, convolved_matrix, filename)

            file_counter += 1

    print("Done")

if __name__ == '__main__':
    main()
