import numpy as np
#################################
class Layer_Dense:
    def __init__(self):
        self.grad_X = None
        self.grad_W = []
        self.grad_b = None
        self.rng = np.random.default_rng()
        self.inputs = None


class Activation_Softmax(Layer_Dense):
    def __init__(self, n_inputs, n_neurons , type):
        super().__init__()
        self.weightslinear = self.rng.normal(0, np.sqrt(2 / (n_inputs + n_neurons)), size=(n_neurons, n_inputs))
        self.weights = self.rng.normal(0, np.sqrt(2 / (n_inputs + n_neurons)), size=(n_neurons, n_inputs))
        if type == 1 :
            self.biases = self.rng.uniform(-1 / np.sqrt(n_inputs), 1 / np.sqrt(n_inputs), size=(n_neurons, 1))
        if type == 0:
            self.data = None
        self.c = None
        self.type = type

    def loss(self, data, c):
        self.c = c
        self.data = data
        l = c.shape[0]
        m = c.shape[1]
        mechane = np.sum(np.exp(np.dot(self.weights, data)), axis=0)
        loss =0

        for k in range(l):
            mone = np.exp(np.dot(data.T, self.weights[k]))
            temp = mone/mechane
            loss = loss + np.dot(c[k], np.log(temp))
        res = -1/m * loss
        return res

    #################################
    def backward(self):
        self.gradient()
        res = self.grad_X
        return res


    def forward(self, inputs):
        if self.type == 0:
            exp_values = np.exp(np.dot( self.weights, inputs) - np.max(np.dot(self.weights, inputs)))
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            return probabilities
        else:
            self.inputs = inputs
            res = np.tanh(np.dot(self.weightslinear, inputs) + self.biases)  # tanh(Wx +b)
            return res

    def Linearbackward(self, v):
        self.grad_W = self.jacobian_test_transpose(self.inputs, v, "weights")
        self.grad_b = self.jacobian_test_transpose(self.inputs, v, "biases")
        res = self.jacobian_test_transpose(self.inputs, v, "inputs")
        return res

    #################################
    def gradient(self):
        self.grad_W =[]
        l = self.c.shape[0]
        m = self.c.shape[1]
        mechane = np.sum(np.exp(np.dot( self.weights, self.data)), axis=0)
        for p in range(l):
            mone = np.exp(np.dot(self.data.T, self.weights[p]))
            res = np.dot(self.data, ((mone/mechane) - self.c[p]))
            res = res / m
            self.grad_W.extend([res])
        mone2 = np.exp(np.dot(self.weights, self.data))
        res2 = (mone2/mechane) - self.c
        res2 = np.dot(self.weights.T, res2)/m
        self.grad_X = res2

    #################################
    def deriv(self, input):
        return 1 - (np.tanh(input) ** 2)

    def jacobian_test(self, x,  v , case):
        nigzeret = np.diag(np.ravel(Activation_Softmax.deriv(self, np.dot(self.weightslinear, x) + self.biases)))
        if(case == "weights"):
            return np.dot(np.dot(nigzeret, np.kron(x.T, np.eye(nigzeret.shape[0]))), np.expand_dims(np.ravel(v, order="F"), axis=1))
        elif(case == "biases"):
            return np.dot(nigzeret, v)
        elif(case == "inputs"):
            return np.dot(np.dot(nigzeret, self.weightslinear), v)



    def jacobian_test_transpose(self, x, v, case):
        wxb = np.dot(self.weightslinear, x) + self.biases
        nigzeret = Activation_Softmax.deriv(self, wxb)
        if(case == "weights"):
            xt = x.T
            return np.dot(nigzeret * v, xt)
        elif(case == "biases"):
            return np.sum(nigzeret * v, axis=1, keepdims=True)
        elif(case == "inputs"):
            return np.dot(self.weightslinear.T, nigzeret * v)


#################################
class SGD:
    def __init__(self, learn_rate, network):
        self.leran_rate = learn_rate
        self.network = network

    def step_size(self):
        for Layer_Dense in self.network.layers:
            step = self.leran_rate * Layer_Dense.grad_W
            new_weight = Layer_Dense.weightslinear - step
            Layer_Dense.weightslinear = new_weight
            step2 = self.leran_rate * Layer_Dense.grad_b
            new_bias = Layer_Dense.biases - step2
            Layer_Dense.biases = new_bias

        arr = np.array(self.network.last_layer.grad_W)
        step =  self.leran_rate * arr
        new_weight = self.network.last_layer.weights - step
        self.network.last_layer.weights = new_weight

#################################

