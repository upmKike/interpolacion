import math as math

import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import Derivative, Poly, Symbol, diff, sympify


class Hermite_Interpolation:

  '''

  Clase para el cálculo del polinomio de Hermite y su evaluación en cualquier
  punto siguiendo el método matricial y de diferencias divididas.

  Parametros
  ----------
  values : list
    Lista de tuplas que contiene como valores los puntos así como  el valor de
    las derivadas en dicho punto, siendo estas las restricciones que debe
    satisfacer el polinomio buscado.

  Atributos
  ----------
  values : list
    Aquí se almacena el argumento values.
  x : np.ndarray
      Es un array de únicamente los puntos dados en values.
  n : int
      Es indice del último punto de values, empezando por 0.
  m : int
      Es el orden de la derivada de mayor orden de la que se tiene información.
  degree : int
      Es el grado que tendrá el polinomio final.
  base_polynomial : sympy.core.add.Add
      Es un polinomio genérico de grado self.degree.
  hermite_polynomial : sympy.core.add.Add
      Es el polinomio de Hermite que cumple las restricciones.

    '''
  def __init__(self, values: list):
    self.values = values
    self.x = np.array([val[0] for val in values])
    self.n = len(values) - 1
    self.m = len(max(self.values,key=len)) - 2
    self.degree = sum([len(ele) - 1 for ele in self.values]) - 1
    print('Degree: ', self.degree)
    self.base_polynomial = sympify(''.join(['+x**' + str(d) \
                                            for d in range(self.degree + 1)]))
    self.hermite_polynomial = self.Hermite_expression(0)


  def coefficients(self, derivative_order: int) -> list:
    '''

    Función auxiliar que devuelve los coeficientes de la derivada de orden
    derivative_order de un polinomio genérico de grado self.degree.

    Parametros
    ----------
    derivative_order : int
      Orden de la derivada de la cual se quieren obtener los coeficientes.

    Devuelve
    -------
    coefficients : list
      Lista con los coeficientes de la derivada del orden pedido de un polinomio genérico.

    '''
    coefficients = Poly(diff(self.base_polynomial,Symbol('x'), derivative_order)).coeffs()

    return coefficients


  def Hermite_matrix(self) -> np.ndarray:
    '''

    Función que crea el sistema de matrices asociado a las condiciones que debe
    cumplir el polinomio de Hermite y devuelve el valor de sus coeficientes.

    Devuelve
    -------
    coefficients : np.ndarray
      Array con los coeficientes del polinomio de Hermite deseado.

    '''
    sum_for_repetitions = 0
    deriv_info = []

    for val in self.values:
      sum_for_repetitions += (len(val) - 1)
      deriv_info = deriv_info + list(val[1:])

    A = []
    for i, x in enumerate(self.values):
      aux = len(x) - 1

      for j in range(aux):
        coefs = self.coefficients(j)

        while len(coefs) != self.degree + 1:
          coefs.append(0)

        for k in range(len(coefs)):
          coefs[k] *= (x[0]**(abs(k - self.degree + j)))

        A.append(coefs)

    A = np.array(A, dtype=np.float64)
    b = np.array(deriv_info)
    coefficients = np.linalg.solve(A, b)

    return coefficients


  def eval_Hermite_from_matrix(self, x_list: np.ndarray, derivative_order=0) -> np.ndarray:
    '''

    Función que recibe una lista de valores x_list y los evalúa en la derivada
    de orden derivative_order del polinomio de Hermite obtenido a través del
    método matricial.

    Parametros
    ----------
    x_list : np.ndarray
      Lista de puntos en los que se quiere evaluar el polinomio de Hermite.
    derivative_order : int
      Orden de derivada del polinomio de Hermite en la que se desean evaluar los puntos

    Devuelve
    -------
    values : np.ndarray
      Array con las imágenes de los puntos evaluados en el polinomio de Hermite deseado.

    '''
    if derivative_order != 0:
      coefficients = self.Hermite_matrix().tolist()

      for i in range(self.degree + 1):
        aux = abs(i - self.degree)
        acum = 1
        j = 0

        while j < derivative_order:
          acum *= (aux - j)
          j += 1
        coefficients[i] *= acum

      for i in range(derivative_order):
        coefficients.pop()
        coefficients = [0] + coefficients

    else:
      coefficients = self.Hermite_matrix()

    values = np.array([sum([(x**(abs(exp  - (self.degree)))) * coefficients[exp] \
                          for exp in range(self.degree + 1)]) for x in x_list])

    return values


  def Hermite_divided_differences(self) -> np.ndarray:
    '''

    Función que utiliza el método de diferencias divididas de Newton extendido
    para nodos repetidos para obtener la pirámide de diferencias divididas
    que contiene al polinomio de Hermite que cumple las restricciones.

    Devuelve
    -------
    x_repeated : np.ndarray
      Array con los nodos repetidos utilizados en las diferencias divididas.
    pyramid : np.ndarray
      Array con la representación en arrays de numpy de la pirámide de diferecias divididas.

    '''
    x_repeated = []
    y = []

    for val in self.values:
      x_repeated = x_repeated + [val[0]] * (len(val) - 1)
      y = y + [val[1]] * (len(val) - 1)

    x_repeated = np.array(x_repeated, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    aux = len(x_repeated)
    pyramid = np.zeros([aux, aux])
    pyramid[:, 0] = y

    for j in range(1,aux):

        for i in range(aux-j):
          if (x_repeated[i+j]-x_repeated[i]) == 0:
            value = None

            for element in self.values:
              if element[0] == x_repeated[i]:
                value = element[j + 1] / math.factorial(j)
                break

            pyramid[i][j] = value

          else:
            pyramid[i][j] = \
           (pyramid[i+1][j-1] - pyramid[i][j-1]) / (x_repeated[i+j]-x_repeated[i])

    return x_repeated, pyramid

  def eval_Hermite_divided_differences(self, x_list: np.ndarray) -> np.ndarray:
    '''

    Función que recibe una lista de valores x_list y los evalúa en
    el polinomio de Hermite obtenido a través del método de diferencias divididas,
    siguiendo el último paso de este algoritmo con los valores de la diagonal
    superior de la pirámide.

    Parametros
    ----------
    x_list : np.ndarray
      Lista de puntos en los que se quiere evaluar el polinomio de Hermite.

    Devuelve
    -------
    final : np.ndarray
      Array con las imágenes de los puntos evaluados en el polinomio de Hermite correspondiente.

    '''
    x_repeated, pyramid = self.Hermite_divided_differences()
    final = []

    for x in x_list:
      res = 0

      for j in range(len(x_repeated)):
        xs = np.array(x_repeated[:j])
        xs = np.unique(xs, return_counts=True)
        value_in_table = pyramid[0][j]

        for i in range(len(xs[0])):
          value_in_table *= ((x - xs[0][i])**(xs[1][i]))
        res += value_in_table
      final.append(res)

    return np.array(final)


  def Hermite_expression(self, derivative_order=0) -> sympy.core.add.Add:
    '''

    Función que crea una expresión de tipo sympy de la derivada de orden derivative_order
    del polinomio de Hermite obtenido para su evaluación y visualización.

    Parametros
    ----------
    derivative_order : int
      Orden de derivada del polinomio de Hermite del que se quiere obtener la expresión

    Devuelve
    -------
    expression : sympy.core.add.Add
      Expresión de derivada de orden elegido del polinomio de Hermite.

    '''
    if derivative_order != 0:
      coefficients = self.Hermite_matrix().tolist()

      for i in range(self.degree + 1):
        aux = abs(i - self.degree)
        acum = 1
        j = 0

        while j < derivative_order:
          acum *= (aux - j)
          j += 1
        coefficients[i] *= acum

      for i in range(derivative_order):
        coefficients.pop()
        coefficients = [0] + coefficients

    else:
      coefficients = self.Hermite_matrix()

    expression = sympify(''.join(['+' + '(' + str(coefficients[d]) \
                            + ')' + '*x**' + str(abs(d - self.degree)) for d in range(self.degree + 1)]))

    return expression


  def plot(self):
    '''

    Función que representa visualmente el polinomio de Hermite y sus derivadas
    hasta el orden máximo del que se posee información, resaltando en cada
    representación los puntos obtenidos en la entrada de la declaración.

    '''
    x_list = np.linspace(min(self.x) - 0.1, max(self.x) + 0.1, 1000)

    print('Representación visual del polinomio de Hermite obtenido y sus derivadas en los puntos dados:\n')


    for derivative in range(self.m + 1):
      plt.title(''.join(['H'] + ["'"]*derivative + ['(x) = ' + str(self.Hermite_expression(derivative))]))
      plt.xlabel('x')
      plt.ylabel(''.join(['y'] + ["'"] * derivative))
      plt.plot(x_list, self.eval_Hermite_from_matrix(x_list, derivative), c='#eba0a0')
      plt.plot([x[0] for x in self.values if len(x) >= (derivative + 2)], [val[derivative + 1] for val in self.values if len(val) >= (derivative + 2)],"o")

      for val in self.values:
        try:
          plt.text(val[0], val[derivative + 1] ,'(' + str(val[0]) + ', ' + \
                  str(val[derivative + 1]) + ')', horizontalalignment='left', \
                  verticalalignment='bottom')
        except:
          continue
      plt.show()
    print('\n--------------------------------------------------------------------------------------------------------\n')

# tests
# ex = Hermite_Interpolation([(8.3, 4, 0), (8.6, 5, 0), (8.9, 6, 0), (9.2, 7, 0)])
# a, b = ex.Hermite_divided_differences()
# print(ex.hermite_polynomial)
# print(a.T)
# print(b)
# ex.plot()
#
# h1 = Hermite_Interpolation([(1,2,1), (3,5,-1)])
# h1.plot()
# h2 = Hermite_Interpolation([(-1, 2, -8, 56), (0, 1, 0, 0), (1, 2, 8, 56)])
# h2.plot()
# h3 = Hermite_Interpolation([(0, -1, -2), (1, 0, 10, 40)])
# h3.plot()
#
# help(Hermite_Interpolation)