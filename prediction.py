from numpy import *
import lasagne
import theano
import theano.tensor as T
import tqdm

SPLIT = 0.7
class Prediction:
    def __init__(self, data, targets, log=False):
        sp = int(SPLIT*len(data))
        # training, test = data[sp:], data[:sp]

        n_inputs = data.shape[1]
        n_targets = max(targets)+1
        print(n_inputs, int(n_targets))

        train_scale = abs(data).max(0)
        train_shift = data.mean(0) / train_scale

        normalized = (data/train_scale - train_shift)[:, None]

        train_scaled, test_scaled = normalized[:sp], normalized[sp:]
        train_targets, test_targets = targets[:sp], targets[sp:]

        input_var = T.row('X', dtype='float64')
        target_var = T.vector('y', dtype='int64')

        network = lasagne.layers.InputLayer((1, n_inputs), input_var)

        network = lasagne.layers.DenseLayer(network,
                                            100,
                                             W=lasagne.init.GlorotUniform(),
                                            nonlinearity = lasagne.nonlinearities.rectify)

        network = lasagne.layers.DenseLayer(network,
                                            n_targets,
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
        self.train_targets = train_targets
        self.test_targets = test_targets
        self.log = log

    
    def train(self):
        for epoch in tqdm.tqdm(range(2000)):
            loss = array([0., 0.])
            for input_batch, target_batch in zip(self.train_scaled, self.train_targets):
                loss += self.train_fn(input_batch, [int(target_batch)])
            if epoch % 200 == 0:
                avg_loss, avg_acc = loss / len(self.train_scaled)
                #print("Epoch %d: Loss %g Acc: %g" % (epoch + 1, avg_loss, avg_acc))
                
    def network_attributes(self):
        return [(self.train_scaled, self.train_targets), (self.test_scaled, self.test_targets), self.predict_fn]
