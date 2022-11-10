import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial


class Data():
    """
    A class to store the function data (domain and codomain) in a Panda Dataframe and calculate the interpolating polynomial and the Vandermonde matrix.
    
    Parameters
    ----------
    x : numpy.array or list
        List of x values/ Function domain 
    y : numpy.array or list
        List of y values / Function codomain
    xlabel : str
        Name of what the domain represents
    ylabel : str
        Name of what the codomain represents
    
    Attributes
    ----------
    x : numpy.array or list
        List of x values/ Function domain 
    y : numpy.array or list
        List of y values / Function codomain
    xlabel : str
        Name of what the domain represents
    ylabel : str
        Name of what the codomain represents
    """
    def __init__(self, x: np.array, y: np.array, xlabel: str = 'No label', ylabel: str = 'No label'):
        """
        Initialice the class

        Parameters
        ----------
        x : numpy.array or list
            List of x values/ Function domain 
        y : numpy.array or list
            List of y values / Function codomain
        xlabel : str
            Name of what the domain represents
        ylabel : str
            Name of what the codomain represents

        Returns
        -------
        None
        """
        self.data = data = pd.DataFrame({'x': x, 'y': y })
        self.xlabel = xlabel
        self.ylabel = ylabel 

    def __str__(self) -> str:
        """
        Prints the domain and codomain of the function as a Pandas Dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        self.data : str
        """
        return str(self.data)

    def vandermonde(self) -> np.array:
        """
        Returns the Vandermonde matrix.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.vander : numpy.array
        """
        return np.vander(self.data.x, self.data.shape[0], increasing = True)

    def plot(self, width: int = 12, height: int = 10):
        """
        Plot the domain and codomain of the function.

        Parameters
        ----------
        width : int
            Default is 12
            Width of the graph
        height: int
            Default is 12
            Height of the graph

        Returns
        -------
        None
        """      
        plt.figure(figsize=(width, height))
        plt.scatter(self.data.x, self.data.y, color = 'orange')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.style.use('seaborn-whitegrid')
        plt.legend()
        plt.show()
    
    def __aux_inter_plot(self, x, y, width: int, height: int, lowlim: int, suplim: int):
        """
        Private function. Do not use
        """ 
        plt.figure(figsize=(width, height))
        plt.ylim(lowlim, suplim)
        plt.scatter(self.data.x, self.data.y, color = 'orange',label = 'Data')
        plt.plot(x, y, color = 'darkred', label = 'Interpolation')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.style.use('seaborn-whitegrid')
        plt.legend()
        plt.show()     

    def interpolation_vandermonde(self, show: bool = False, width: int =  12, height: int = 10, precision: float = 0.1, lowlim: int = 0, suplim: int = 10) -> Polynomial:
        """
        Returns the interpolating polynomial using the Valdemort Matrix. The polynomial is represented using the Numpy Polynomial class.
        Calculates the Valdemort Matrix and solves the system of linear equations using the Numpy library.

        Parameters
        ----------
        show : bool
            Default is False
            If show is True it represent the polynomial in a graph.
        width : int
            Default is 12
            Width of the graph
        height: int
            Default is 12
            Height of the graph
        precision: float 
            Default is 0.1
            Precision of the graph. How big is the gap between each number of the graph.
        lowlim: int = 0 
            Default is 0
            Lower limit of the y-axis of the graph
        suplim: int = 10
            Default is 10
            Superior limit of the y-axis of the graph

        Returns
        -------
        numpy.Polinomial : numpy.array
            Polinomial represent in a numpy Polinomial object
        """
        vander = np.vander(self.data.x, self.data.shape[0], increasing = True)
        coef = np.linalg.solve(vander, self.data.y)
        pol = Polynomial(coef)
        if show == True:
            x = np.arange(self.data["x"].min(), self.data["x"].max() + precision, precision)
            y = pol(x)
            self.__aux_inter_plot(x, y, width, height, lowlim, suplim)
        return pol 

    def __aux_lagrange(self, x, y, vec):
        """
        Private function. Do not use
        """ 
        aux = Polynomial(y)
        for a in vec:
            if x != a:
                aux *= Polynomial([-a/(x-a), 1/(x-a)])
        return aux

    def interpolation_lagrange(self, show: bool = False, width: int = 12, height: int = 10, precision: float = 0.1, lowlim: int = 0, suplim: int = 10) -> Polynomial:
        """
        Returns the interpolating polynomial using the Lagrange Method. The polynomial is represented using the Numpy Polynomial class. 

        Parameters
        ----------
        show : bool
            Default is False
            If show is True it represent the polynomial in a graph.
        width : int
            Default is 12
            Width of the graph
        height: int
            Default is 12
            Height of the graph
        precision: float 
            Default is 0.1
            Precision of the graph. How big is the gap between each number of the graph.
        lowlim: int = 0 
            Default is 0
            Lower limit of the y-axis of the graph
        suplim: int = 10
            Default is 10
            Superior limit of the y-axis of the graph

        Returns
        -------
        numpy.Polinomial : numpy.array
            Polinomial represent in a numpy Polinomial object
        """
        x = self.data["x"].values.copy();    y = self.data["y"].values.copy()
        pol = Polynomial(0)
        for i in range(len(x)):
            pol += self.__aux_lagrange(x[i], y[i], x)
        if show == True:
            x = np.arange(self.data["x"].min(), self.data["x"].max() + precision, precision)
            y = pol(x)
            self.__aux_inter_plot(x, y, width, height, lowlim, suplim)
        return pol

    def __aux_newton(self, x, y):
        """
        Private function. Do not use
        """ 
        aux = Polynomial(y)
        for a in x:
            aux *= Polynomial([-a, 1])
        return aux

    def interpolation_newton(self, show: bool = False, width: int = 12, height: int = 10, precision: float = 0.1, lowlim: int = 0, suplim: int = 10)-> Polynomial:
        """
        Returns the interpolating polynomial using the Newton Method. The polynomial is represented using the Numpy Polynomial class. 

        Parameters
        ----------
        show : bool
            Default is False
            If show is True it represent the polynomial in a graph.
        width : int
            Default is 12
            Width of the graph
        height: int
            Default is 12
            Height of the graph
        precision: float 
            Default is 0.1
            Precision of the graph. How big is the gap between each number of the graph.
        lowlim: int = 0 
            Default is 0
            Lower limit of the y-axis of the graph
        suplim: int = 10
            Default is 10
            Superior limit of the y-axis of the graph

        Returns
        -------
        numpy.Polinomial : numpy.array
            Polinomial represent in a numpy Polinomial object
        """
        x = self.data["x"].values.copy();    y = self.data["y"].values.copy()
        m = len(x)
        for k in range(1, m):
            y[k:m] = (y[k:m] - y[k - 1])/(x[k:m] - x[k - 1])
        pol = Polynomial(0)
        for i in range(m):
            pol += self.__aux_newton(x[:i], y[i])
        if show == True:
            x = np.arange(self.data["x"].min(), self.data["x"].max() + precision, precision)
            y = pol(x)
            self.__aux_inter_plot(x, y, width, height, lowlim, suplim)
        return pol

def error_interpolation(f: Polynomial, inter: Polynomial, x: np.array) -> np.array:
    """
    Calculates the error between a function and the estimation of the interpolation polynomial  given a point cloud

    Parameters
    ----------
    f: numpy.Polynomial
        Function (correct values)
    inter: numpy.Polynomial
        Interpolation (estimation values)
    x: numpy.array
        Point cloud where the error is to be calculated
        
    Returns
    -------
    Values : numpy.array
        Error in each point
    """
    return abs(f(x) - inter(x))

def error_interpolation_points(y: np.array, inter: Polynomial, x: np.array) -> np.array:
    """
    Calculates the error between the codomain of the function and the estimation of the interpolation polynomial

    Parameters
    ----------
    f: numpy.array
        Codomain of the function (correct values)
    inter: numpy.Polynomial
        Interpolation (estimation values)
    x: numpy.array
        Point cloud where the error is to be calculated 

    Returns
    -------
    values : numpy.array
        Error in each point
    """
    return abs(y - inter(x))

if __name__ == "__main__":
    
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0.31, 0.32, 0.33, 0.34])

    d1 = Data(x, y)
    d1.plot()
    a1 = d1.interpolation_vandermonde(True, suplim = 0.5)
    a2 = d1.interpolation_lagrange(True, suplim = 0.5)
    a3 = d1.interpolation_newton(True, suplim = 0.5)

    print(a1)
    print(a2)
    print(a3)
    print(a1(0.4))
    print(a2(0.4))
    print(a3(0.4))

    e1 = error_interpolation_points(y,a1 ,x)
    e2 = error_interpolation_points(y,a2 ,x)
    e3 = error_interpolation_points(y,a3 ,x)

    plt.plot(x,e1, label = 'Error Interpolación Vandermonde')
    plt.plot(x,e2, label = 'Error Interpolación Lagrange')
    plt.plot(x,e3, label = 'Error Interpolación Newton')
    plt.legend()
    plt.show() 


