import numpy

from algorithms.linear_curve_fitter import LinearCurveFitter

if __name__ == '__main__':
    fitter = LinearCurveFitter()

    scenario_data = numpy.recfromcsv('sample_data/housing_prices.csv', delimiter=',', filling_values=numpy.nan, case_sensitive=True, deletechars='', replace_space=' ')
    area = scenario_data['area']
    num_bedrooms = scenario_data['num_bedrooms']
    price = scenario_data['price']
    x = numpy.vstack([area, num_bedrooms]).transpose()
    fitter.fit_data(x, price.reshape(price.shape[0], 1))
    print("Price of a 1650sqft house with 3 bedrooms is: " +
          str(fitter.evaluate(numpy.array([2000, 4]))))