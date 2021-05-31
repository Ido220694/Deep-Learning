from softmax import SGD
from Neural_network import NeuralNetwork
from train import Train
from scipy.io import loadmat
from plotter import Plotter

if __name__ == '__main__':
    # dataset = loadmat("SwissRollData.mat")
    dataset = loadmat("GMMData.mat")
    #dataset = loadmat("PeaksData.mat")
    Yt = dataset["Yt"]
    Ct = dataset["Ct"]
    Yv = dataset["Yv"]
    Cv = dataset["Cv"]
    p = Plotter(10, 10, Yt, Ct)
    plotter1 = p.gradient_test_plot()
    network = NeuralNetwork(Yt.shape[0], Ct.shape[0], 10, 5)
    trainer = Train(network, Yt, Ct, Yv, Cv, 32, 15, SGD(0.1, network))
    results = trainer.training()
    plotter2 = p.plot_accuracy(results)
    plotter3 = p.plot_loss(results)
    plotter4 = p.plot_jacobian_tests()
    plotter5 = p.gradient_test_plot2()
    plotter6 = p.plot_accuracy(results)
    plotter7 = p.plot_loss(results)


