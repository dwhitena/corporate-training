import matplotlib.pyplot as plt
import numpy as np

def logistic(x):
    """
    Compute the logistic function with input x.

    Args:
        x - logistic function input
    Returns:
        1/(1 + e^-x)
    """
    return 1/(1 + np.exp(-x))

def main():
    
    # sample some points from the logistic function
    x = np.arange(-6, 6, 0.001)
    y = logistic(x)

    # plot the points using matplotlib
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('logistic(x)')
    plt.show()

if __name__ == "__main__":
    main()
