import numpy


def minimize_with_gradient_descent(f, gradient, start_point, optional_args=None):
    """Implement the gradient descent algorithm to minimize the a specified
    function

    Keyword arguments:
    f -- a callable function that represents the function whose
                value is to be minimized. This function will be invoked with
                jval(theta_vec, x_input, y_input) where theta_vec consists
                of the coefficients to be varied
    gradient -- a callable function that represents the function that
                    calculates the derivative of jval
    start_point -- a numpy.ndarray with the initial values for each coefficient in
                   function (if None, a zero array will be assumed)
    optional_args -- a dictionary containing one or more of the following
                        optional arguments
        max_num_iterations -- maximum number of iterations for which this algorithm
                              will be run
        alpha -- factor by which gradient is multiplied before decrementing the product
                       from theta at every turn
    Return:
    A list where the elements at respective indices are:
        0 -- Boolean indicating whether the gradient descent converged successfully
        1 -- numpy.ndarray containing coefficients obtained as a result of
                successful convergence. None in case convergence failed.
        2 -- numpy.ndarray showing the value of jval evaluated at each step. In case
                of successful convergence, the size of this array will equal the number
                of iterations. In case convergence fails, the size of this array will
                equal the maximum number of iterations at which the gradient descent was
                terminated
    """
    max_num_iterations = 1000
    alpha = 0.01
    if optional_args:
        if "alpha" in optional_args.keys():
            if not isinstance(optional_args["alpha"], float):
                raise ValueError("Optional argument alpha must be a float")
            alpha = optional_args["alpha"]
        if "max_num_iterations" in optional_args.keys():
            if not isinstance(optional_args["max_num_iterations"], int):
                raise ValueError("Optional argument max_num_iterations must be an integer")
            max_num_iterations = optional_args["max_num_iterations"]

    # Initialize an array of theta. This will represent the values of theta
    # which we will iteratively modify
    theta = start_point

    # Keep a counter keeping track of iteration count
    iter_count = 0

    # Keep a flag keeping track of whether the gradient descent converged
    converged = False

    # Pre-initialize an array to hold the value of function output at each
    # iteration. If we actually do manage convergence, then we will trim
    # it down to iter_count in size
    function_evals = numpy.zeros((max_num_iterations, 1))

    # Initialize with the first function eval
    function_evals[0] = f(theta)

    while iter_count < max_num_iterations - 1:
        iter_count += 1
        prev_theta = theta
        gradient_result = gradient(prev_theta)
        theta = prev_theta - gradient_result * alpha
        function_evals[iter_count] = f(theta)
        if function_evals[iter_count] - function_evals[iter_count - 1] < 0.001:
            converged = True
            #break

    return [converged, theta, function_evals, iter_count]
