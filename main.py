#test
import src.int_hermite

values = [(-1, 2, -8, 56), (0, 1, 0, 0), (1, 2, 8, 56)] 
print('The degree of the polynomial will be ' + str((len(values)) * (len(values[0]) - 1) - 1))
x, pyramid = src.int_hermite.divided_diff(values)
print(x)
print(pyramid)