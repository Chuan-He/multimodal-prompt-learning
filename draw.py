import matplotlib.pyplot as plt
import numpy as np
import argparse


def main():
    x = [2,3,4,5,6]
    y_base = [72.53,72.79,72.8,72.6,73.03]
    y_new = [65.75,65.17,66.55,64.86,66.46]
    fig, ax = plt.subplots()
    #plt.plot(x, y_base)
    ax.plot(x, y_base, label='linear') # 作y1 = x 图，并标记此线名为linear
    
    ax.plot(x, y_new, label='quadratic') #作y2 = x^2 图，并标记此线名为quadratic
    ax.set_xlabel('number of prompts') #设置x轴名称 x label
    ax.set_ylabel('accuracy(%)') #设置y轴名称 y label
    ax.set_title('Number of prompts') #设置图名为Simple Plot
    ax.legend() #自动检测要在图例中显示的元素，并且显示

    #plt.xlabel('time (s)')
    #plt.ylabel('voltage (mV)')
    #plt.title('About as simple as it gets, folks')
    ax.grid(True)
    plt.savefig("test.png")
    plt.show()
    print(1)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("directory", type=str, help="path to directory")
    #args = parser.parse_args()

    main()
