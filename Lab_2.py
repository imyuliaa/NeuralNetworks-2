import math
import matplotlib.pyplot as plt
import numpy as np


def print_vec(vec):
    size = int(math.sqrt(len(vec)))
    for i in range(0, size):
        for j in range(0, size):
            if vec[i * size + j] == 1:
                print('+', end='')
            else:
                print('-', end='')
        print()


def activation_func(s, T):
    if s <= 0:
        return 0
    elif s > T:
        return T
    else:
        return s


def max_val(vec):
    print('\n\nProbabilities per test templates:')
    for i in range(0, len(vec)):
        print(str(i + 1) + ' -> ', vec[i])

    return vec.index(max(vec))


patterns = {
    '0': [
        -1, 1, 1, -1,
        1, -1, -1, 1,
        1, -1, -1, 1,
        -1, 1, 1, -1,
    ],
    '1': [
        1, -1, -1, 1,
        1, 1, -1, 1,
        1, -1, 1, 1,
        1, -1, -1, 1,
    ],
}

pattern_count = len(patterns)
pattern_size = len(patterns['0'])
k = 1 / pattern_count
T = k * pattern_size

print('Training data:')
for i in patterns:
    reshaped_pattern = np.reshape(patterns[i], [4, 4])
    plt.subplot(2, 5, int(i)+1)
    plt.axis('off')
    plt.title('learning '+i)
    plt.imshow(reshaped_pattern, cmap='gray_r')
plt.show()


first_layer = [[] for i in range(0, pattern_count)]

print('First layer weights:')
for i in patterns:
    for j in range(pattern_size):
        first_layer[int(i)].append(k * patterns[i][j])
        print(first_layer[int(i)][j], end=' ')
    print()

eps = 0.3

test_vectors = {
   '0': [
        -1, 1, 1, -1,
         1, 1, 1, 1,
        -1, 1, 1, 1,
        -1, 1, 1, -1,
    ],
   '1': [
        1, -1, -1, 1,
        1, -1, -1, 1,
        1, -1, -1, 1,
        -1, 1, 1, -1,
    ],
    '2': [
        1, 1, 1, 1,
        -1, 1, -1, -1,
        -1, -1, 1, -1,
        1, 1, 1, 1,
    ],
}

for tvi in test_vectors:
    test_vector = test_vectors[tvi]

    means = []
    for i in patterns:
        temp = 0
        for j in range(pattern_size):
            temp = temp + test_vector[j] * first_layer[int(i)][j]
        means.append(temp)
        print('Mean ['+i+']:' + str(means[int(i)]))

    print('\nSecond layer weights:')
    second_layer = [[] for i in range(0, pattern_size)]
    for i in range(pattern_count):
        for j in range(pattern_count):
            if i == j:
                second_layer[i].append(1)
            else:
                second_layer[i].append(-1 * eps)
            print(second_layer[i][j], end='\t')
        print()

    last_iter = [means[i] for i in range(pattern_count)]
    next_iter = [0 for i in range(pattern_count)]
    diff = [0 for i in range(pattern_count)]

    reshaped_test_vector = np.reshape(test_vector, [4, 4])
    plt.axis('off')
    plt.title('Test vector '+tvi)
    plt.imshow(reshaped_test_vector, cmap='gray_r')
    plt.show()

    it = 0
    while True:
        norm = 0
        for i in range(pattern_count):
            temp = 0
            for j in range(pattern_count):
                temp += last_iter[j] * second_layer[i][j]
            next_iter[i] = activation_func(temp, T)
        diff[i] = next_iter[i] - last_iter[i]

        for i in range(pattern_count):
            last_iter[i] = next_iter[i]
            norm = norm + diff[i] ** 2

        if math.sqrt(norm) <= eps or it == 1000:
            break
        it = it + 1
    print('\nTest vector is similar to ' + str(max_val(next_iter)+1) + ' pattern')

