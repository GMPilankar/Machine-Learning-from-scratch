# Linear Regression from Scratch

This repository contains a custom implementation of a **Linear Regression** model written in Python using the **NumPy** library. It also includes a unit test suite to verify its correctness by comparing its output against `scikit-learn`'s `LinearRegression` model.

---

## Features

* **Closed-Form Solution**: The model uses the **Normal Equation** and the **pseudo-inverse** to find the optimal weights and bias in a single step, without the need for iterative methods like gradient descent.
* **Highly Optimized**: The core calculations are performed using efficient NumPy functions for matrix operations, ensuring fast performance.
* **Interoperable**: The model's `fit` and `predict` methods are designed to be compatible with `scikit-learn`'s API for easy integration.

---

## Files

* `linear_regression.py`: Contains the `LinearReg` class with `fit` and `predict` methods.
* `test.py`: A unit test file that validates the `LinearReg` class. It uses a standard regression dataset from `scikit-learn` and compares the results to the `sklearn.linear_model.LinearRegression` class.

---

## How to Run the Tests

To verify that the custom implementation works correctly, you can run the unit tests from the command line.

1.  Clone this repository.
2.  Make sure you have the necessary libraries installed:
    ```bash
    pip install numpy scikit-learn
    ```
3.  Navigate to the directory and run the test file:
    ```bash
    python -m unittest test.py
    ```

If the tests pass, the output will indicate a successful run. This confirms that your `LinearReg` class accurately computes the same weights and predictions as the industry-standard `scikit-learn` library.