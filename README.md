Repositorio común con todos los módulos juntos.

Para usarlo, simplemente instálalo usando "pip". Por ejemplo en colab:
        !pip install git+https://github.com/upmKike/interpolacion.git

Y luego puedes usarlo directamente, por ejemplo:

        import interpolacion.src.int_hermite as hermite

        values = [(-1, 2, -8, 56), (0, 1, 0, 0), (1, 2, 8, 56)] 
        print('The degree of the polynomial will be ' + str((len(values)) * (len(values[0]) - 1) - 1))
        x, pyramid = hermite.divided_diff(values)
        print(x)
        print(pyramid)
