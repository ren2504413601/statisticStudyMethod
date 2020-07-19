# encoding=utf-8
# author : renlei
from matplotlib import pyplot as plt
import numpy as np
def load_data(file_name):
    locs, prices = [], []
    with open(file_name,'r') as f:
        for sample in f.readlines():
            loc, price = sample.strip().split(',')
            locs.append(loc)
            prices.append(price)
        f.close()
        locs, prices = np.array(locs, dtype= np.float), \
                 np.array(prices, dtype = np.float)
    return (locs-locs.mean())/locs.std(), prices
def visualize(x, y):
    plt.figure()
    plt.scatter(x, y, c = 'b')
    # plt.show()
    return 
# Get regression model under LSE criterion with degree 'deg'
def get_model(x, y, deg = 3):
    p = np.polyfit(x, y, deg)
    return lambda input_x: np.polyval(p, input_x)
def get_loss(x, y, deg = 3):
    y_pre = get_model(x, y, deg)(x)
    return 0.5*((y - y_pre)**2).sum()

if __name__ == "__main__":
    locs, prices = load_data("./_Data/prices.txt")
    visualize(x = locs, y = prices)
    plt.figure()
    locs_test = np.linspace(-2, 4, 100)
    for d in [1, 3, 5]:
        pprice = get_model(locs, prices, deg=d)(locs_test)
        plt.plot(locs_test, pprice, label="degree = {}".format(d))
        plt.legend()
        print("d = {}, mean square loss between true value and regression predicted value ={}".format(
            d, get_loss(locs, prices, deg = d)
        )
        )
    plt.show()