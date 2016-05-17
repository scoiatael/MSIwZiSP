from numpy import *
import lasagne
import theano
import theano.tensor as T
import tqdm

def normalize(arr):
    a = []
    arr = list(map(lambda x: x[1], arr))
    arr = list(zip(*[arr[i:] for i in range(4)]))
    for i in range(len(arr)):
        foo = []
        foo += list(arr[i][:-1])
        diff = arr[i][-1] - arr[i][-2]
        foo += [sign(diff)]
        a.append(foo)
    return array(a)[:, None]

class Prediction:
    def __init__(self, data, log=False):
        training, test = map(normalize, data)

        train_scale = abs(training).max(0)
        train_scale[0, -1] = 1

        train_shift = training.mean(0) / train_scale
        train_shift[0, -1] = -1

        train_scaled, test_scaled = map(lambda x: x/train_scale - train_shift, [training, test])

        input_var = T.row('X', dtype='float64')
        target_var = T.vector('y', dtype='int64')

        network = lasagne.layers.InputLayer((1,3), input_var)

        network = lasagne.layers.DenseLayer(network,
                                            100,
                                             W=lasagne.init.GlorotUniform(),
                                            nonlinearity = lasagne.nonlinearities.rectify)
        network = lasagne.layers.DenseLayer(network,
                                            3,
                                            nonlinearity = lasagne.nonlinearities.softmax)

        # create loss function
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
                network, lasagne.regularization.l2)

        # create parameter update expressions
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01,
                                                    momentum=0.9)
        acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                              dtype=theano.config.floatX)

        # compile training function that updates parameters and returns training loss
        train_fn = theano.function([input_var, target_var], [loss, acc], updates=updates)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
        
        self.predict_fn = predict_fn
        self.train_fn = train_fn
        self.train_scaled = train_scaled
        self.test_scaled = test_scaled
        self.log = log

    
    def train(self):
        for epoch in tqdm.tqdm(range(2000)):
            loss = array([0., 0.])
            for batch in self.train_scaled:
                input_batch = batch[:, :-1]
                target_batch = batch[:, -1]
                loss += self.train_fn(input_batch, [round(target_batch[0])])
            if epoch % 200 == 0:
                avg_loss, avg_acc = loss / len(self.train_scaled)
                #print("Epoch %d: Loss %g Acc: %g" % (epoch + 1, avg_loss, avg_acc))
                
    def network_attributes(self):
        return [self.train_scaled, self.test_scaled, self.predict_fn]