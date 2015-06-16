from algorithms.gradient_descent import *
from algorithms.curve_fitter import CurveFitter


class LinearCurveFitter(CurveFitter):
    """Linear curve fitter that fits a curve of the form
    y = theta_0 + theta_1 * x1 + theta_2 * x2 + ... + theta_n * xn

    Implements a method named fit_data that fits a curve to input data
    by computing an array containing the coefficients:
    [theta_0, theta_1, theta_2, ... theta_n]
    for the input factors [x0, x1, x2, x3, ... , xn] where x0 = 1

    It has another function called evaluate which evaluates the previously
    fit function at the specified input (x0, x1, x2, ... , xn)
    """

    def __init__(self):

        # Vector containing mean of each input factors
        self.mean = numpy.zeros((1, 1))

        # vector containing the standard deviation of input factors
        self.deviation = numpy.zeros((1, 1))

        # Flag keeping track of whether a valid fit for input data was
        # found
        self.valid_fit_found = False

        # Evaluated coefficients for fit curve
        self.coefficients = None

        # Set default values for optional parameters
        self.alpha = 0.01
        self.max_num_iterations = 1000
        self.start_point = None

        # Variables holding scaled and augmented input data
        self.x_input = numpy.zeros((1, 1))
        self.y_input = numpy.zeros((1, 1))

    def fit_data(self, x_input, y_input, optional_args=None):
        """Fits parameters to specified x_input and y_input by
        assuming a cost function (based on the fitter type)

        :param x_input: numpy.ndarray containing sampling points. Number of
                        factors will be the number of columns of this array
                        while number of rows will indicate number of sampling
                        points (each row being one sampling point)
        :param y_input: value of function at sampling points
        :param optional_args: dictionary of optional named arguments, namely:
            max_num_iterations -- maximum number of iterations for which this algorithm
                              will be run
            start_point -- a numpy.ndarray with the initial values for each factor in
                           function (if None, a zero array will be assumed)
            alpha -- factor by which gradient is multiplied before decrementing the product
                           from theta at every turn
        """
        # Validate the input parameters
        self.validate_inputs(x_input, y_input, optional_args)

        # As the first step, we do feature scaling of all the factors
        # that have been input. We also cache the scaling factors used so that
        # we can use that while making predictions
        scaled_x_input = self.scale_factors(x_input)

        # We assume a hypothesis function of the form:
        # h = theta_0 + theta_1 * x1 + theta_2 * x2 + theta_3 * x3 + ... + theta_n * xn

        # We can re-write the above hypothesis function in the form
        # h = theta_0 * x0 + theta_1 * x1 + theta_2 * x2 + theta_3 * x3 + ... + theta_n * xn
        # where x0 = 1
        # so as to be able to take advantage of a vectorised equation form:
        # h = theta_vec' * x_input_augmented

        # In order to obtain the augmented x_input (i.e. x_input with the first column of
        # just 1s), we append a column to x_input
        x0 = numpy.ones((scaled_x_input.shape[0], 1))
        x_input_augmented = numpy.hstack([x0, scaled_x_input])

        self.x_input = x_input_augmented
        self.y_input = y_input

        result = minimize_with_gradient_descent(self.calculate_cost,
                                                self.calculate_gradient,
                                                self.start_point,
                                                optional_args)
        if result[0]:
            self.valid_fit_found = True
            self.coefficients = result[1]

    def evaluate(self, x_input):
        """Evaluates the previously fit function at specified point
        Note that this must only be called after specifying a data to
        fit a curve with (i.e. after calling the fit_data method)

        :param x_input: numpy.ndarray indicating point of evaluation
        :return: evaluated value of function at specified point
        """
        self.validate_inputs(x_input)

        # We need to scale the factors just as we did before performing
        # gradient descent
        scaled_x_input = numpy.subtract(x_input, self.mean)
        scaled_x_input = numpy.divide(scaled_x_input, self.deviation)

        # Augment input x with 1 so that we can use vectorised multiplication
        scaled_x_input = numpy.hstack([[1], scaled_x_input])

        # Evaluate the function
        return numpy.dot(scaled_x_input, self.coefficients)

    def scale_factors(self, x_input):
        """Scales the input factors and caches the mean and standard deviations"""

        self.mean = numpy.mean(x_input, axis=0)
        self.deviation = numpy.std(x_input, axis=0)
        scaled_x_input = numpy.subtract(x_input, self.mean)
        scaled_x_input = numpy.divide(scaled_x_input, self.deviation)
        return scaled_x_input

    def calculate_cost(self, theta):
        """Calculates the cost for a specified theta

        :param theta: value of function at sampling points
        :return: scalar indicating the cost of the function at specified theta
        """

        # Evaluate hypothesis function with given theta
        h = numpy.dot(self.x_input, theta)
        # Evaluate error between hypothesis and actual value
        err = numpy.subtract(self.y_input, h)
        # Calculate cost
        return numpy.square(err).sum() / (2 * self.x_input.shape[0])

    def calculate_gradient(self, theta):
        """Calculates first derivative of cost function for a particular theta

        :param theta: value of function at sampling points
        :return: scalar indicating the cost of the function at specified theta
        """

        # Evaluate hypothesis function with given theta
        h = numpy.dot(self.x_input, theta)
        # Evaluate error between hypothesis and actual value
        err = numpy.subtract(h, self.y_input)

        prod = numpy.multiply(err, self.x_input)
        prod_sum = numpy.sum(prod, axis=0).reshape((theta.shape[0], 1))

        return prod_sum / (self.x_input.shape[0])

    def validate_inputs(self, x_input, y_input=None, optional_args=None):
        if not isinstance(x_input, numpy.ndarray):
            raise ValueError("Input x_input is not of type numpy.array")
        if y_input != None:
            if not isinstance(y_input, numpy.ndarray):
                raise ValueError("Input y_input is not of type numpy.array")
            if x_input.shape[0] != y_input.shape[0]:
                raise ValueError("Number of rows of x_input must match number of rows of y_input")
        else:
            return

        if optional_args:
            if "start_point" in optional_args.keys():
                if not isinstance(optional_args["start_point"], numpy.ndarray):
                    raise ValueError("Optional argument start_point is not of type numpy.array")
                if optional_args["start_point"].shape[0] != (x_input.shape[0] + 1):
                    raise ValueError("Number of rows specified for start_point must equal 1 + number of rows of x_input")
                self.start_point = optional_args["start_point"]
            if "alpha" in optional_args.keys():
                if not isinstance(optional_args["alpha"], float):
                    raise ValueError("Optional argument alpha must be a float")
                self.alpha = optional_args["alpha"]
            if "max_num_iterations" in optional_args.keys():
                if not isinstance(optional_args["max_num_iterations"], int):
                    raise ValueError("Optional argument max_num_iterations must be an integer")
                self.max_num_iterations = optional_args["max_num_iterations"]
        else:
            # We did not pre-initialize start_point with the other optional parameters
            # since we didn't want to allocate a possible large array of zeros if we could
            # avoid it (as would be the case if start_point was specified by user)
            # Hence, we initialize it now
            self.start_point = numpy.zeros((x_input.shape[1]+1, 1))
