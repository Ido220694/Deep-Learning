
import numpy as np
from matplotlib import pyplot as plt

from tests import MainTest



class Plotter:
    def __init__(self, l_layers, hidden_layer_size, input, output):
        super().__init__()
        self.test = MainTest(l_layers, hidden_layer_size, input, output)

    def gradient_test_plot(self):
        res = self.test.gradient_test()
        plt.title("2.1.1 : Gradient Test")
        plt.xlabel("epochs")
        plt.xticks(np.arange(15), np.arange(1, 16))
        plt.plot(res[0], "-o",
                 label = "$|f(x+ \\epsilon d) - f(x)|$")
        plt.plot(res[1], "-o",
                 label = "$|f(x+ \\epsilon d^T) - f(x) - \\epsilon d*grad_x|$")
        plt.legend(loc="upper right")
        plt.show()
        plt.clf()

    def gradient_test_plot2(self):
        res = self.test.gradient_test_wholeNetwork()
        plt.title("2.2.3 : Gradient Test Network")
        plt.xlabel("epochs")
        plt.xticks(np.arange(15), np.arange(1, 16))
        plt.plot(res[0], "-o",
                 label="$|f(x+ \\epsilon d) - f(x)|$")
        plt.plot(res[1], "-o",
                 label="$|f(x+ \\epsilon d^T) - f(x) - \\epsilon d*grad_x|$")
        plt.legend(loc="upper right")
        plt.show()
        plt.clf()



    def plot_jacobian_tests(self):
        results = self.test.jaacobian_test()
        plt.xlabel("epochs")
        plt.title("Jacobian Test")
        plt.xticks(np.arange(15), np.arange(1, 16))
        plt.plot(results[0][0], "-o",
                     label ="respect to w:$|f(x+\\epsilon d) -f(x)|$")
        plt.plot(results[0][1], "-o",
                     label ="respect to w:$|f(x+\\epsilon d) -f(x) -JacMV(x,\\epsilon )|$")
        plt.plot(results[1][0], "-o",
                     label ="respect to b:$|f(x+\\epsilon d) -f(x)|$")
        plt.plot(results[1][1], "-o",
                     label ="respect to b:$|f(x+\\epsilon d) -f(x) -JacMV(x,\\epsilon )|$")

        plt.plot(results[2][0], "-o",
                     label ="respect to x:$|f(x+\\epsilon d) -f(x)|$")
        plt.plot(results[2][1], "-o",
                     label ="respect to x:$|f(x+\\epsilon d) -f(x) -JacMV(x,\\epsilon )|$")

        plt.legend(loc="upper right")
        plt.show()
        plt.clf()

    def plot_jacobiantransposs_tests(self):
        results = self.test.jaacobian_test_transpossed()
        plt.xlabel("epochs")
        plt.title("Jacobian Trasnposse Test")
        plt.xticks(np.arange(15), np.arange(1, 16))
        plt.plot(results[0], "-o",
                     label ="respect to w:$|u^T JacMV(x,v) - v^T JacTMV(x,u)|$")
        plt.plot(results[1], "-o",
                     label ="respect to b:$|u^T JacMV(x,v) - v^T JacTMV(x,u)|$")
        plt.plot(results[2], "-o",
                     label ="respect to x:$|u^T JacMV(x,v) - v^T JacTMV(x,u)|$")

        plt.legend(loc="upper right")
        plt.show()
        plt.clf()
######################################################################################33

    def plot_accuracy(self, training_results):

        train_success = training_results[4]
        validation_success = training_results[5]
        title = ("Training and Validation Accuracy\n"
                  f"Batch size: {training_results[0]}, dim: {training_results[1]}, Layers: {training_results[2]}")

        plt.title(title)
        x = np.arange(len(train_success))
        plt.xticks(x, x + 1)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(train_success, "-o", label="Training")
        plt.plot(validation_success, "-o", label="Validation")
        plt.legend()
        plt.show()
        plt.clf()



    def plot_loss(self, training_results):
        title = ("Loss Function Graph\n"
                f"Batch size: {training_results[0]}, dim: {training_results[1]}, Layers: {training_results[2]} ")

        losses = training_results[3]
        plt.title(title)
        x = np.arange(len(losses))
        plt.xticks(x, x + 1)
        plt.plot(losses, "-o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        plt.clf()
###################################################