from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

# ESTA CELDA ES LA GENERACION DE LOS DATOS DE ENTRADA PARA USAR LAS DIFERENTES FUNCIONES DE INTERPOLACION

# def f(x):
#     return np.exp(-2*x) * np.sin(10*np.pi*x)

#lista_puntos = [] # funcion
#for i in np.linspace(0, 1, 90):
#    lista_puntos.append([i, f(i)])
    
#lista_pred = [] # puntos a interpolar
#lista_pos = np.linspace(0, 1, 15)
#for i in lista_pos:
#    lista_pred.append([i, f(i)])
    
    
#plt.figure(figsize=(10, 6))
#plt.plot(*zip(*lista_puntos), '-')
#plt.plot(*zip(*lista_pred), 'o-')


def formula_lagrange(x: float, x0: float, y0: float, x1: float, y1: float) -> Tuple[float, float]:
    '''
    FORMULA DE LAGRANGE\n
    x0, y0 = punto 1\n 
    x1, y1 = punto 2\n 
    x = valor a interpolar
    '''
    return x, (x - x1) * y0 / (x0 - x1) + (x - x0) * y1 / (x1 - x0)

def formula_lagrange_directa(x: float, lista_pred: List[float], f) -> Tuple[float, float]:
    '''
    FORMULA DE NEWTON-LAGRANGE (verifica si el valor a interpolar esta dentro de los intervalos)\n
    x = valor a interpolar,\n 
    lista_pred = lista de puntos de los que se quiere interpolar (intervalos), es decir, posicion en x
    '''
    lista_pos_x = [i[0] for i in lista_pred]
    lista_pos_x_tuples = [(lista_pos_x[i], lista_pos_x[i+1]) for i in range(len(lista_pos_x)-1)]
    for i in lista_pos_x_tuples:
        if x >= i[0] and x <= i[1]:
            return formula_lagrange(x, i[0], f(i[0]), i[1], f(i[1]))
    return "not found"
    
def formula_newton(x: float, x0: float, y0: float, x1: float, y1: float) -> Tuple:
    '''
    FORMULA DE NEWTON\n
    x0, y0 = punto 1\n 
    x1, y1 = punto 2\n 
    x = valor a interpolar
    '''
    c1 = (y1 - y0) / (x1 - x0)
    c0 = y0
    return x, c0 + c1 * (x - x0)

def formula_newton_directa(x: float, lista_pred: List[float], f) -> Tuple[float, float]:
    '''
    FORMULA DE NEWTON DIRECTA (verifica si el valor a interpolar esta dentro de los intervalos)\n
    x = valor a interpolar,\n 
    lista_pred = lista de puntos de los que se quiere interpolar (intervalos), es decir, posicion en x
    '''
    lista_pos_x = [i[0] for i in lista_pred]
    lista_pos_x_tuples = [(lista_pos_x[i], lista_pos_x[i+1]) for i in range(len(lista_pos_x)-1)]
    for i in lista_pos_x_tuples:
        if x >= i[0] and x <= i[1]:
            return formula_newton(x, i[0], f(i[0]), i[1], f(i[1]))
    return "not found"


# EJEMPLOS DE USO
#tests
#punto_pred_lagrange = formula_lagrange(0.5, lista_pred[4][0], lista_pred[4][1], lista_pred[5][0], lista_pred[5][1])
#punto_pred_newton = formula_newton(0.5, lista_pred[1][0], lista_pred[1][1], lista_pred[2][0], lista_pred[2][1])
#punto_pred_lagrange_directa = formula_lagrange_directa(0.5, lista_pred)
#punto_pred_newton_directa = formula_newton_directa(0.5, lista_pred)
# PODEMOS VER QUE LA DIFERENCIA ENTRE LOS DOS METODOS ES NULA; SU RESULTADO ES EL MISMO.


def plot_interpolacion(lista_pred: List[float] = [], punto_pred: List[float] = [], lista_puntos: List[float] = None) -> None:
    '''
    PLOT DE LA INTERPOLACION\n
    Plotea la funcion y los puntos a interpolar junto con los puntos interpolados
    
    Args:
     * :param lista_pred = lista de puntos a interpolar\n
     * :param punto_pred = lista de puntos interpolados\n
     * :param lista_puntos = lista de puntos de la funcion
    
    '''
    plt.figure(figsize=(10, 6))
    # Ploteo de funcion
    if lista_puntos:
        plt.plot(*zip(*lista_puntos), '-', linewidth=2, label="Funcion")
        
    # Ploteo de puntos a interpolar
    plt.plot(*zip(*lista_pred), 'o', linewidth=2, label="Puntos a interpolar")
    
    # Ploteo de puntos interpolados
    if len(punto_pred) == 2:
        plt.plot(*punto_pred, '--', linewidth=2, color='green', label="Puntos interpolados")
    else:
        plt.plot(*zip(*punto_pred), '--', linewidth=2, color='green', label="Puntos interpolados")
    
    # Lineas dashed verticales
    for pos in lista_pred: 
        plt.vlines(pos[0], 0, pos[1], linestyles='dashed', color = 'orange')
        
    # Linea horizontal en 0
    plt.hlines(0, lista_pred[0][0], lista_pred[-1][0], colors='black')
    
    # Grid
    plt.grid()
    
    # Leyenda
    plt.legend()
    
#test
# plot_interpolacion(lista_pred, punto_pred_lagrange_directa, lista_puntos)


def interpolacion_one_time(x, lista_pos_x: List[float] = None, lista_pos_y: List[float] = None, tipo = 'slinear', lista_pred: List[tuple] = None) -> List[float]:
    '''
    INTERPOLACION DE UNA SOLA VEZ\n
    Args:
        * :param x = valor a interpolar\n
        * :param lista_pos_x = lista de posiciones en x\n
        * :param lista_pos_y = lista de posiciones en y\n
        * :param tipo = tipo de interpolacion ( "slinear", "quadratic", "cubic" )\n
        * :param lista_puntos = lista de puntos de la funcion\n
        
    Returns:
        * :return lista de puntos interpolados\n    
    '''
    if tipo != "cubic":
    
        if lista_pred:
            lista_pos_x = [i[0] for i in lista_pred]
            lista_pos_y = [i[1] for i in lista_pred]
            return [x, interp1d(lista_pos_x, lista_pos_y, kind = tipo)(x)]
        
        return [x, interp1d(lista_pos_x, lista_pos_y, kind = tipo)(x)]
    
    else:
        if lista_pred:
            lista_pos_x = [i[0] for i in lista_pred]
            lista_pos_y = [i[1] for i in lista_pred]
            funcion_cubic = CubicSpline(lista_pos_x, lista_pos_y, bc_type='natural')
            return [x, funcion_cubic(x)]
        
        funcion_cubic = CubicSpline(lista_pos_x, lista_pos_y, bc_type='natural')
        return [x, funcion_cubic(x)]

#test
#interpolacion_one_time(0.5, lista_pred= lista_pred, tipo='cubic')


def general_interpolacion_select(lista_pos_x: List[float] = [], lista_pos_y: List[float] = [], lista_pred: List[tuple] = [], tipo: str = 'linear') -> List[float]:
    '''
    CALCULO DE LA INTERPOLACION\n
    Calcula la interpolacion en los itervalos dados.
    
    Args:
    * :param x: punto a predecir\n
    * :param lista_pos_x: lista de puntos x (opcional_1)\n
    * :param lista_pos_y: lista de puntos y (opcional_1)\n
    * :param lista_pred: lista de puntos (x, y) (opcional_2)\n
    * :param tipo: tipo de interpolacion ( "slinear", "quadratic", "cubic" )\n
    
    Returns:
    * :return: lista de puntos interpolados. X e Y por separado.
    '''
    
    if tipo != "cubic":
        
        if len(lista_pos_x) + len(lista_pos_y) >= 2:
            x_lins = np.linspace(lista_pos_x[0], lista_pos_x[-1], 1000)
            return [x_lins, interp1d(lista_pos_x, lista_pos_y, kind = tipo)(x_lins)]
        
        lista_pos_x = [i[0] for i in lista_pred]
        lista_pos_y = [i[1] for i in lista_pred]
        
        x_lins = np.linspace(lista_pos_x[0], lista_pos_x[-1], 1000)
        return [x_lins, interp1d(lista_pos_x, lista_pos_y, kind = tipo)(x_lins)]
    
    else:
        
        
        if len(lista_pos_x) + len(lista_pos_y) >= 2:
            x_lins = np.linspace(lista_pos_x[0], lista_pos_x[-1], 1000)
            funcion_cubic = CubicSpline(lista_pos_x, lista_pos_y, bc_type="natural")
            return [x_lins, funcion_cubic(x_lins)]
        
        lista_pos_x = [i[0] for i in lista_pred]
        lista_pos_y = [i[1] for i in lista_pred]
        
        x_lins = np.linspace(lista_pos_x[0], lista_pos_x[-1], 1000)
        funcion_cubic = CubicSpline(lista_pos_x, lista_pos_y, bc_type="natural")
        return [x_lins, funcion_cubic(x_lins)]


#test
   
#lista_pos_x = lista_pos.copy()
#lista_pos_y = [f(i) for i in lista_pos_x]
#
## EJEMPLO DE USO
#interpolacion_pred = general_interpolacion_select(lista_pos_x, lista_pos_y, tipo = 'cubic') # lista_puntos = lista_pred
#plot_interpolacion(lista_pred, interpolacion_pred) # lista_puntos = lista_puntos (opcional para visualizar la funcion original)


def interpolate_and_plot(lista_pos_x: List[float] = None, lista_pos_y: List[float] = None, lista_pred: List[Tuple] = None, tipo: str = 'slinear') -> List[float]:
    '''
    CALCULO DE LA INTERPOLACION + PLOT\n
    Calcula la interpolacion en los itervalos dados y los plotea.
    
    Args:
    * :param lista_pos_x: lista de puntos x (opcional_1)\n
    * :param lista_pos_y: lista de puntos y (opcional_1)\n
    * :param lista_pred: lista de puntos (x, y) (opcional_2)\n
    * :param tipo: tipo de interpolacion ( "slinear", "quadratic", "cubic" )\n
    * :param visualizar_funcion: visualizar la funcion interpolada\n
    
    Returns:
    * :return: lista de puntos interpolados
    '''
    if lista_pos_x is not None and lista_pos_y is not None:
        lista_pred = list(zip(lista_pos_x, lista_pos_y))
        interpolacion_pred = general_interpolacion_select(lista_pred= lista_pred, tipo = tipo)
        plot_interpolacion(lista_pred, interpolacion_pred)
        return list(zip(interpolacion_pred[0], interpolacion_pred[1]))
    else:
        interpolacion_pred = general_interpolacion_select(lista_pred= lista_pred, tipo = tipo)
        plot_interpolacion(lista_pred, interpolacion_pred)
        return list(zip(interpolacion_pred[0], interpolacion_pred[1]))

#test
# EJEMPLO DE USO 
#interpolate_and_plot(lista_pos_x, lista_pos_y, tipo = 'cubic')
#interpolate_and_plot(lista_pred = lista_pred, tipo = 'cubic')    


def cubic_spline(y_list: list, x_list: list, x: float=None, return_fun_coefficient=False):
    """
    CALCULO DE LA INTERPOLACION DEL SPLINE CUBICO
    Calculo de manera manual, mediante implementacion matematica directa

    Args:
    * :param y_list: lista de puntos y
    * :param x_list: lista de puntos x
    * :param x: punto x a predecir (opcional)
    * :param return_fun_coefficient: para devolver los coeficientes de la funcion cubica obtenida (por defecto Falso)

    Returns:
    * return: if return_fun_coefficient == True, devuelve los coeficientes de la funcion cubica obtenida
    """
    if x != None:
        i = 0
        while x_list[i] < x:
            i += 1
        x_complete = x_list[:]
        y_complete = y_list[:]
        x_list = x_list[i-1:i+2]  
        y_list = y_list[i-1:i+2]
    
    len_y_list = len(y_list)
    A = np.diag([2]*len_y_list).astype("int8")
    A[1:-1, 1:-1] *= 2
    for i in range(len_y_list):
        try:
            A[i, i+1] += 1
            A[i, i-1] += 1
        except:
            try:
                A[i, i+1] += 1
            except:
                A[i, i-1] += 1

    b = np.array([[0]*len_y_list]).reshape(len_y_list, -1)
    b[0,:] = 3 * (y_list[1] - y_list[0])
    for i in range(2, len_y_list):
        b[i-1,:] = 3 * (y_list[i] - y_list[i-2])
    b[len_y_list - 1,:] = 3 * (y_list[-1] - y_list[-2])

    D = np.linalg.solve(A, b).reshape(-1,)
  
    c = np.empty(len_y_list - 1)
    d = np.empty(len_y_list - 1)
    for i in range(len_y_list - 1):
        c[i] = 3 * (y_list[i+1] - y_list[i]) - 2 * D[i] - D[i+1]
        d[i] = 2 * (y_list[i] - y_list[i+1]) + D[i] + D[i+1]
    var_dict = {"a": y_list, "b": D, "c": c, "d": d}
  
    plt.figure(figsize=(15, 10))

    if x != None:
        t = (x - x_list[0]) / (x_list[1] - x_list[0])
    else:
        t = np.linspace(0, 1, 100)
        X = []
        Y = []
        for i in range(len_y_list - 1):
            X.extend(np.linspace(x_list[i], x_list[i+1], 100))
            for val in t:
                Y.append(var_dict["a"][i] + var_dict["b"][i] * val + var_dict["c"][i] * val**2 + var_dict["d"][i] * val**3)
        plt.plot(X, Y)
        plt.plot(x_list, y_list, marker='o', linestyle="None", color="red")

    if x != None:
        y = var_dict["a"][0] + var_dict["b"][0] * t + var_dict["c"][0] * t**2 + var_dict["d"][0] * t**3
        plt.plot(x_complete, y_complete, marker='o', linestyle="None", color="red")
        plt.plot(x, y, marker='o', linestyle="None", color="green")
  
    if return_fun_coefficient:
        try:
            print(f'Para x = {x}, obtenemos la función Y(t) = {var_dict["a"][0]} + {var_dict["b"][0]}*t + {var_dict["c"][0]}*t^2 + {var_dict["d"][0]}*t^3.\nPara este caso, t = {t}, por lo que Y({t}) = {y}')
            return y
        except:
            return var_dict
    else:
        try:
            return y
        except:
            pass