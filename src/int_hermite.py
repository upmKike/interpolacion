import math as math

import matplotlib.pyplot as plt
import numpy as np
from sympy import Derivative, Poly, Symbol, diff, sympify


def divided_diff(values):
    '''
    function to calculate the divided
    differences table
    '''
    x_initial = np.array([element[0] for element in values])
    x = np.repeat(x_initial, len(values[0]) - 1)
    y_initial= np.array([element[1] for element in values])
    y = np.repeat(y_initial, len(values[0]) - 1)
    n = len(y)
    coef = np.zeros([n, n])
    # the first column is y
    coef[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            if (x[i+j]-x[i]) == 0:
                value = None
                for element in values:
                    if element[0] == x[i]:
                        value = element[j + 1] / math.factorial(j)
                        break
                coef[i][j] = value
            else:
                coef[i][j] = \
               (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
            
    return x, coef

#test
# values = [(-1, 2, -8, 56), (0, 1, 0, 0), (1, 2, 8, 56)] 
# print('The degree of the polynomial will be ' + str((len(values)) * (len(values[0]) - 1) - 1))
# x, pyramid = divided_diff(values)
# print(x)
# print(pyramid)


def Hermite(x, aux, dd_table): # Hay qu emodificar para devolver coeficientes
    res = 0
    for j in range(len(aux)):
        xs = np.array(aux[:j])
        xs = np.unique(xs, return_counts=True)
        value_in_table = dd_table[0][j]
        for i in range(len(xs[0])):
            value_in_table *= ((x - xs[0][i])**(xs[1][i]))
        res += value_in_table
    return res

def Hermite_expression(coefficients, degree, d_order):
    i = degree
    coef = coefficients[::-1]
    final_str = ''
    while i >= 0:
        if coef[i] > 0:
            final_str = final_str + '+' + str(coef[i]) + '*x**' + str(i)
        elif coef[i] == 0:
            continue
        else:
            final_str = final_str + str(coef[i]) + '*x**' + str(i)
        i -= 1 
    x = Symbol('x')
    expr = sympify(final_str)
    deriv = diff(expr, x, d_order)
    return deriv

#tests
#---------------------------------------------------------

# aux, dd_table = divided_diff(values)
# x = np.linspace(-1.1,1.1, 1000)
# y_hermite = np.array([Hermite(xi, aux, dd_table) for xi in x])
# 
# plt.plot(x, y_hermite)
# plt.plot([val[0] for val in values], [val[1] for val in values],"o")
# for val in values:
#   plt.text(val[0], val[1] ,'(' + str(val[0]) + ', ' + str(val[1]) + ')', horizontalalignment='left', verticalalignment='bottom')
# plt.show()
# 
# x_vals = np.linspace(0,4, 1000)
# y = [eval_Hermite(xi, Hermite_matrixA , Hermite_matrixB) for xi in x_vals]
# plt.plot(x_vals, y)
# plt.plot([val[0] for val in values], [val[1] for val in values],"o")
# for val in values:
#   plt.text(val[0], val[1] ,'(' + str(val[0]) + ', ' + str(val[1]) + ')', horizontalalignment='left', verticalalignment='bottom')
# plt.show()
# 
# deriv1 = Hermite_expression(aux, len(aux) - 1, 1)
# print(deriv1)
# x_vals = np.linspace(0,4, 1000)
# y = [deriv1.evalf(subs={x: r}) for r in x_vals]
# plt.plot(x_vals, y)
# plt.plot([val[0] for val in values], [val[2] for val in values],"o")
# for val in values:
#   plt.text(val[0], val[2] ,'(' + str(val[0]) + ', ' + str(val[2]) + ')', horizontalalignment='left', verticalalignment='bottom')
# plt.show()
#


def derivative_H(n, polynomial):
    x = Symbol('x')
    expr = sympify(polynomial)
    deriv = diff(expr, x, n)
    a = Poly(deriv, x)
    a.coeffs()
    return a.coeffs()

def gen_polynomial(degree):
    polynomial = ''
    d = 0 
    while d <= degree:
        polynomial = polynomial + '+x**' + str(d)
        d += 1
    return polynomial

def Hermite_matrix(values):
    n = len(values) - 1
    m = len(values[0]) - 2
    degree = (n + 1) * (m + 1) - 1
    polynomial = gen_polynomial(degree)

    A = np.zeros(((n+1) * (m+1), degree + 1))
    b = np.zeros(((n+1) * (m+1)))

    for i in range(len(A)):
        b[i] = values[(i // (m + 1))][i % (m + 1) + 1]
        coefs = derivative_H(i % (len(values[0]) - 1), polynomial)
        for j in range(len(coefs) - 1):
            coefs[j] *= ((values[(i // (m + 1))][0])**(abs(j - degree) - (i % (m + 1))))
        while len(coefs) != len(A[0]):
            coefs.append(0.)
        A[i] = coefs
    return A, b

def eval_Hermite(x, A, b):
    coef = np.linalg.solve(A, b)
    res = 0
    for exp in range(len(coef)):
        res += ((x**(abs(exp - (len(A)-1)))) * coef[exp])
    return res 


def Hermite_expression2(coefficients, degree, d_order):
    i = degree
    coef = coefficients[::-1]
    final_str = ''
    while i >= 0:
        if coef[i] > 0:
            final_str = final_str + '+' + str(coef[i]) + '*x**' + str(i)
        elif coef[i] == 0:
            continue
        else:
            final_str = final_str + str(coef[i]) + '*x**' + str(i)
        i -= 1 
    x = Symbol('x')
    expr = sympify(final_str)
    deriv = diff(expr, x, d_order)
    return deriv

#tests
# values = [(1,2,1), (3,5,-1)]
# print('The degree of the polynomial will be ' + str((len(values)) * (len(values[0]) - 1) - 1))     
# Hermite_matrixA , Hermite_matrixB = Hermite_matrix(values)
# coef = np.linalg.solve(Hermite_matrixA , Hermite_matrixB)
# 
# x_vals = np.linspace(0,4, 1000)
# y = [eval_Hermite(xi, Hermite_matrixA , Hermite_matrixB) for xi in x_vals]
# plt.plot(x_vals, y)
# plt.plot([val[0] for val in values], [val[1] for val in values],"o")
# for val in values:
#   plt.text(val[0], val[1] ,'(' + str(val[0]) + ', ' + str(val[1]) + ')', horizontalalignment='left', verticalalignment='bottom')
# plt.show()
# 
# deriv1 = Hermite_expression(coef, len(coef) - 1, 1)
# x_vals = np.linspace(0,4, 1000)
# y = [deriv1.evalf(subs={x: r}) for r in x_vals]
# plt.plot(x_vals, y)
# plt.plot([val[0] for val in values], [val[2] for val in values],"o")
# for val in values:
#   plt.text(val[0], val[2] ,'(' + str(val[0]) + ', ' + str(val[2]) + ')', horizontalalignment='left', verticalalignment='bottom')
# plt.show()