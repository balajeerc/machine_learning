from algorithms.curve_fitter import CurveFitter


class LinearCurveFitter(CurveFitter):

    def find_fitting_parameters(self, x_input, y_input, optional_args=None):
        """
        Fits parameters to specified x_input and y_input by
        assuming a cost function (based on the fitter type)

        :param x_input: numpy.ndarray containing sampling points. Number of
                        factors will be the number of columns of this array
        :param y_input: value of function at sampling points
        :param optional_args: dictionary of optional named arguments
        :return: numpy.ndarray containing the list of coefficients
                 that best fit the input data
        """
        raise NotImplementedError("Should have implemented this")