class CurveFitter(object):
    """
    Interface that different curve fitters conform to. Various
    curve fitters are defined in this project such as:
    i) LinearRegressionFitter
    ii) PolynomialRegressionFitter
    """

    def fit_data(self, x_input, y_input, optional_args=None):
        """
        Fits parameters to specified x_input and y_input by
        assuming a cost function (based on the fitter type)

        :param x_input: numpy.ndarray containing sampling points
        :param y_input: value of function at sampling points
        :param optional_args: dictionary of optional named arguments
        :return: numpy.ndarray containing the list of coefficients
                 that best fit the input data
        """
        raise NotImplementedError("Should have implemented this")

    def evaluate(self, x_input):
        """
        Evaluates the previously fit function at specified point

        :param x_input: numpy.ndarray indicating point of evaluation
        :return: evaluated value of function at specified point
        """
        raise NotImplementedError("Should have implemented this")