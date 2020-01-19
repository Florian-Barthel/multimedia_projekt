import numpy as np
print('step')
array = np.array([[[1, 2, 3, 4], [4, 5, 5, 4]], [[1, 2, 3, 2], [4, 5, 5, 4]]])
print(array)

print('step')
indices = np.argmin(np.sum(array, axis=-1))
print(indices)

print('step')
actual = np.take(array, indices, axis=-2)
print(actual)
