#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

'''
Testing a multiple linear regression in Numpy.
'''

# System Libraries.
import mysql.connector
import math
import numpy
import random



def multipleLinearRegression():
    ''' Execute a multiple linear regression. '''

    # First value is always '1' to include a constant term or intersect.
    xs = [[1,1,1,1], [1,2,1,1], [1,1,2,1], [1,1,1,2], [1,2,2,1], [1,2,1,2], [1,1,2,2], [1,2,2,2], [1,3,2,2], [1,2,3,2], [1,2,2,3]]

    # Build one y with a bif of noise.
    random.seed()
    ys = []
    for x in xs:
        # The 1 could be x[0] but 1 makes the point that this column is always 1 for a constant term.
        y = 10 + x[1] + 2*x[2] + 3*x[3] + random.random()
        ys.append(y)

    # Fit a multiple linear regression.
    p, residuals, rank, s = numpy.linalg.lstsq(xs, ys, rcond=None)

    # This is the fit coefficients.
    print(f'coefficients = {p}')

    # Show the resulting model.
    print(f'(0,0,0) => {p[0]:,.1f}  (10) ')
    for x in xs:
        calculatedValue = p[0] + p[1] * x[1] + p[2] * x[2] + p[3] * x[3]
        print(f'({x[1]},{x[2]},{x[3]}) => {calculatedValue:,.1f}  ({10 + x[1] + 2*x[2] + 3*x[3]:,.0f})')



if __name__ == '__main__':
    multipleLinearRegression()
