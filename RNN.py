# Copyright (c) 2012-2013, Razvan Pascanu
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
RNN model for training on recurrent neural models.

Author: Razvan Pascanu
contact : r.pascanu@gmail

Details:
    * I use `omega` for the value of the regularization term and `alpha` for
    the factor of the regularization term
    * rho measures the spectral radius of the recurrent weight matrix


"""

## Trick for flushing stdout (if script is used with tee)
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self,data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

import sys
sys.stdout=Unbuffered(sys.stdout)
## end trick

import numpy, time
import theano
import theano.tensor as TT

from tempOrder import TempOrderTask
from addition import AddTask
from memorization import MemTask
from multiplication import MulTask
from permutation import PermTask
from tempOrder3bit import TempOrder3bitTask


def jobman(state, channel):
    ###### CONSTRUCT DATASET
    if 'bound' not in state:
        state['bound'] = 1e-20
    if 'minerr' not in state:
        state['minerr'] = .01
    if 'l2' not in state:
        state['l2'] = 0
    floatX = theano.config.floatX
    if channel is not None:
        channel.save()
    n_hidden = state['nhid']
    rng = numpy.random.RandomState(state['seed'])
    max_val =state['cutoff']
    if state['task'] == 'torder':
        task = TempOrderTask(rng, floatX)
    elif state['task'] == 'torder3':
        task = TempOrder3bitTask(rng, floatX)
    elif state['task'] == 'perm':
        task = PermTask(rng, floatX)
    elif state['task'] == 'mul':
        task = MulTask(rng, floatX)
    elif state['task'] == 'add':
        task = AddTask(rng, floatX)
    elif state['task'] == 'mem':
        task = MemTask(rng, floatX, state['memvalues'], state['mempos'],
                       state['memall'])
    nin = task.nin
    nout = task.nout

    ########## INITIALIZE PARAMS
    if state['init'] == 'sigmoid':
        W_uh = numpy.asarray(
            rng.normal(size=(nin, n_hidden), scale= .01, loc = .0), dtype = floatX)
        W_hh = numpy.asarray(
            rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = floatX)
        W_hy = numpy.asarray(
            rng.normal(size=(n_hidden, nout), scale =.01, loc=0.0), dtype = floatX)
        b_hh = numpy.zeros((n_hidden,), dtype=floatX)
        b_hy = numpy.zeros((nout,), dtype=floatX)
        activ = TT.nnet.sigmoid
    elif state['init'] == 'test':
        W_uh = numpy.asarray(
            rng.normal(size=(nin, n_hidden), scale= 8e-1, loc = .0), dtype = floatX)
        W_hh = numpy.asarray(
            rng.normal(size=(n_hidden, n_hidden), scale=8e-1, loc = .0), dtype = floatX)
        W_hy = numpy.asarray(
            rng.normal(size=(n_hidden, nout), scale =8e-1, loc=0.0), dtype = floatX)
        b_hh = numpy.zeros((n_hidden,), dtype=floatX)
        b_hy = numpy.zeros((nout,), dtype=floatX)
        activ = lambda x: x
    elif state['init'] == 'basic_tanh':
        W_uh = numpy.asarray(
            rng.normal(size=(nin, n_hidden), scale= .1, loc = .0), dtype = floatX)
        W_hh = numpy.asarray(
            rng.normal(size=(n_hidden, n_hidden), scale=.1, loc = .0), dtype = floatX)
        W_hy = numpy.asarray(
            rng.normal(size=(n_hidden, nout), scale =.1, loc=0.0), dtype = floatX)
        b_hh = numpy.zeros((n_hidden,), dtype=floatX)
        b_hy = numpy.zeros((nout,), dtype=floatX)
        activ = TT.tanh
    elif state['init'] == 'smart_tanh':
        W_uh = numpy.asarray(
            rng.normal(size=(nin, n_hidden), scale= .01, loc = .0), dtype = floatX)
        W_hh = numpy.asarray(
            rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = floatX)
        for dx in xrange(n_hidden):
            spng = rng.permutation(n_hidden)
            W_hh[dx][spng[15:]] = 0.
        sr = numpy.max(abs(numpy.linalg.eigvals(W_hh)))
        W_hh = numpy.float32(.95* W_hh/sr)

        W_hy = numpy.asarray(
            rng.normal(size=(n_hidden, nout), loc=0.0, scale = .01), dtype = floatX)
        b_hh = numpy.zeros((n_hidden,), dtype=floatX)
        b_hy = numpy.zeros((nout,), dtype=floatX)
        activ = TT.tanh


    W_uh = theano.shared(W_uh, 'W_uh')
    W_hh = theano.shared(W_hh, 'W_hh')
    W_hy = theano.shared(W_hy, 'W_hy')
    b_hh = theano.shared(b_hh, 'b_hh')
    b_hy = theano.shared(b_hy, 'b_hy')

    ########### DEFINE TRAINING FUNCTION
    u = TT.tensor3()
    t = TT.matrix()
    # Regularization term factor
    alpha = TT.scalar()
    lr = TT.scalar()
    h0_tm1 = TT.alloc(numpy.array(0, dtype=theano.config.floatX), state['bs'], n_hidden)

    def recurrent_fn(u_t, h_tm1, W_hh, W_uh, W_hy):
        h_t = activ(TT.dot(h_tm1, W_hh) + TT.dot(u_t, W_uh) + b_hh)
        return h_t

    h, _ = theano.scan(recurrent_fn, sequences = u,
                       outputs_info = [h0_tm1],
                       non_sequences = [W_hh, W_uh, W_hy],
                       name = 'recurrent_fn',
                       mode = theano.Mode(linker='cvm'))
    # Trick to get dC/dh[k]
    scan_node = h.owner.inputs[0].owner
    assert isinstance(scan_node.op, theano.scan_module.scan_op.Scan)
    n_pos = scan_node.op.n_seqs + 1
    init_h = scan_node.inputs[n_pos]
    y = TT.nnet.softmax(TT.dot(h[-1], W_hy) + b_hy)
    cost = -(t * TT.log(y)).mean(axis=0).sum()

    # Compute gradients
    gW_hh, gW_uh, gW_hy,\
           gb_hh, gb_hy, gH, g_on_H = TT.grad(
               cost, [W_hh, W_uh, W_hy, b_hh, b_hy, init_h, h])

    initial_gWhh = TT.zeros_like(W_hh)
    d_ht = TT.tensor3('dht')
    d_on_ht = TT.tensor3('d_on_ht')
    ht = TT.tensor3('ht')
    # d_ht[i]    = (d+ c/d h[i]) + \sum_k>0 (d c/d h[i+k])(d h[i+k])/(d h[i])
    # d_on_ht[i] = (d+ c/d h[i])
    if 'sigmoid' in state['init']:
        tmp_x = d_ht[1:] * ht * (1-ht)
    elif 'tanh' in state['init']:
        tmp_x = d_ht[1:] * (1 - ht**2)
    else:
        tmp_x = d_ht[1:]
    sh0 = tmp_x.shape[0]
    sh1 = tmp_x.shape[1]
    sh2 = tmp_x.shape[2]
    tmp_x = tmp_x.reshape((sh0*sh1, sh2))
    tmp_x = TT.dot(tmp_x, W_hh.T)
    tmp_x = (tmp_x.reshape((sh0, sh1, sh2))**2).sum(2)
    tmp_y = (d_ht[1:]**2).sum(2)
    tmp_reg = (TT.switch(TT.ge(tmp_y, state['bound']), tmp_x/tmp_y, 1) -1.)**2
    n_elems = TT.mean(TT.ge(tmp_y, state['bound']),axis=1)
    tmp_reg = tmp_reg.mean(1).sum()/n_elems.sum()
    tmp_gWhh = TT.grad(tmp_reg, W_hh)
    [tmp_reg, tmp_gWhh, n_elems] = theano.clone([tmp_reg, tmp_gWhh,
                                                 n_elems.mean()],
                       replace=[(d_ht, gH),
                                (d_on_ht, g_on_H),
                                (ht,h)])
    if state['alpha'] > 0:
        gW_hh = gW_hh + tmp_gWhh * alpha

    norm_theta = TT.sqrt((gW_hh**2).sum() +
                         (gW_uh**2).sum() +
                         (gW_hy**2).sum() +
                         (gb_hh**2).sum() +
                         (gb_hy**2).sum() )
    if state['clipstyle'] == 'rescale':
        c = state['cutoff']
        gW_hh = TT.switch(norm_theta > c, c*gW_hh/norm_theta, gW_hh)
        gW_uh = TT.switch(norm_theta > c, c*gW_uh/norm_theta, gW_uh)
        gW_hy = TT.switch(norm_theta > c, c*gW_hy/norm_theta, gW_hy)
        gb_hh = TT.switch(norm_theta > c, c*gb_hh/norm_theta, gb_hh)
        gb_hy = TT.switch(norm_theta > c, c*gb_hy/norm_theta, gb_hy)
        # due to numerical precision issues in float32 we assume that we can
        # not even trust the numbers we get if the following `new_cond` is
        # true
        new_cond = TT.or_(TT.or_(TT.isnan(norm_theta),
                                 TT.isinf(norm_theta)),
                          TT.or_(norm_theta < 0,
                                 norm_theta > 1e10))

        gW_hh = TT.switch(new_cond, numpy.float32(.02)*W_hh, gW_hh)
        gW_uh = TT.switch(new_cond, numpy.float32(.0), gW_uh)
        gW_hy = TT.switch(new_cond, numpy.float32(.0), gW_hy)
        gb_hh = TT.switch(new_cond, numpy.float32(.0), gb_hh)
        gb_hy = TT.switch(new_cond, numpy.float32(.0), gb_hy)

    train_step = theano.function([u,t, alpha, lr],[cost, norm_theta,
                                                   tmp_reg, n_elems],
                            on_unused_input='warn',
                            updates=[(W_hh, W_hh - lr*gW_hh),
                                     (W_uh, W_uh - lr*gW_uh),
                                     (W_hy, W_hy - lr*gW_hy),
                                     (b_hh, b_hh - lr*gb_hh),
                                     (b_hy, b_hy - lr*gb_hy)])


    u = TT.tensor3()
    t = TT.matrix()
    h0_tm1 = TT.alloc(numpy.array(0, dtype=theano.config.floatX), state['cbs'], n_hidden)

    def recurrent_fn(u_t, h_tm1, W_hh, W_uh, W_hy):
        h_t = activ(TT.dot(h_tm1, W_hh) + TT.dot(u_t, W_uh) + b_hh)
        return h_t

    h, _ = theano.scan(recurrent_fn, sequences = u,
                       outputs_info = [h0_tm1],
                       non_sequences = [W_hh, W_uh, W_hy],
                       name = 'validation_recurrent_fn',
                       mode = theano.Mode(linker='cvm'))
    if task.classifType == 'lastSoftmax':
        y = TT.nnet.softmax(TT.dot(h[-1], W_hy) + b_hy)
        cost = -(t * TT.log(y)).mean(axis=0).sum()
        error = TT.neq(TT.argmax(y, axis=1), TT.argmax(t, axis=1)).mean()
    elif task.classifType == 'softmax':
        nwh = h.reshape((h.shape[0]*h.shape[1], h.shape[2]))
        y = TT.nnet.softmax(TT.dot(nwh, W_hy) + b_hy)
        cost = -(t * TT.log(y)).mean(axis=0).sum()

        if task.report == 'all':
            nwy = y.reshape((h.shape[0], h.shape[1], b_hy.shape[0])).argmax(2)
            nwt = t.reshape((h.shape[0], h.shape[1], b_hy.shape[0])).argmax(2)
            error = (TT.neq(nwy, nwt).sum(0) > 0).mean()
        else:
            nwy = y.reshape((h.shape[0], h.shape[1], t.shape[1]))
            nwt = t.reshape((h.shape[0], h.shape[1], t.shape[1]))
            error = TT.neq(TT.argmax(nwy[-1], axis=1), TT.argmax(nwt[-1],
                                                                axis=1)).mean()
    elif task.classifType == 'lastLinear':
        y = TT.dot(h[-1], W_hy) + b_hy
        cost = ((t - y)**2).mean(axis=0).sum()
        error = (((t - y)**2).sum(axis=1) > .04).mean()

    eval_step = theano.function([u,t], [cost, error])

    print 'Starting to train'
    best_score = 100
    cont = True
    n = -1
    solved = 0
    state['solved'] =0
    avg_cost = 0
    avg_norm = 0
    avg_reg = 0
    avg_steps = 0
    avg_len = 0
    avg_time = 0
    alpha = state['alpha']
    lr = state['lr']
    store_space = state['maxiters'] // state['checkFreq']
    store_train = numpy.zeros((state['maxiters'],), dtype='float32') -1
    store_valid = numpy.zeros((store_space,), dtype='float32') -1
    store_norm = numpy.zeros((state['maxiters'],), dtype='float32') -1
    store_rho = numpy.zeros((store_space,), dtype='float32') -1
    store_reg = numpy.zeros((state['maxiters'],), dtype='float32') -1
    store_steps = numpy.zeros((state['maxiters'],), dtype='float32') -1

    last_save = time.time()
    max_length = state['max_length']
    min_length = state['min_length']
    while lr > 1e-8 and cont and n<state['maxiters']:
        n = n+1
        if max_length > min_length:
            length = min_length + rng.randint(max_length - min_length)
        else:
            length = min_length
        train_x, train_y = task.generate(state['bs'], length)

        st = time.time()
        tr_cost, norm_theta, tmp_reg, tnelems = train_step(train_x, train_y, alpha,
                                                  lr)
        ed = time.time()
        avg_cost += tr_cost
        store_train[n] = tr_cost
        store_norm[n] = norm_theta
        store_reg[n] = tmp_reg
        store_steps[n] = tnelems

        avg_norm += norm_theta
        avg_reg += tmp_reg
        avg_steps += tnelems
        avg_len += length
        avg_time += (ed - st)

        if n % state['checkFreq'] == 0 and n > 0:
            avg_cost = avg_cost / float(state['checkFreq'])
            avg_norm = avg_norm / float(state['checkFreq'])
            avg_reg = avg_reg / float(state['checkFreq'])
            avg_steps = avg_steps / float(state['checkFreq'])
            avg_len = avg_len / float(state['checkFreq'])
            avg_time = avg_time
            valid_cost = 0
            error = 0
            for dx in xrange(state['ebs'] // state['cbs']):
                if max_length > min_length:
                    length = min_length + rng.randint(max_length - min_length)
                else:
                    length = min_length
                valid_x, valid_y = task.generate(state['cbs'], length)
                _cost, _error = eval_step(valid_x, valid_y)
                valid_cost = valid_cost + _cost
                error = error + _error
            valid_cost = valid_cost / float(state['ebs']//state['cbs'])
            error = error*100. / float(state['ebs']//state['cbs'])
            rho =numpy.max(abs(numpy.linalg.eigvals(W_hh.get_value())))
            print 'Iter %07d'%n,':',\
                    'train nnl %05.3f, ' % avg_cost,\
                    'valid error %07.3f%%, '% error,\
                    'best valid error %07.3f%%, '% best_score,\
                    'average gradient norm %7.3f, '% avg_norm, \
                    'rho_Whh %5.2f, '% rho, \
                    'Omega %5.2f, '% float(avg_reg), \
                    'alpha %6.3f, ' % alpha, \
                    'steps in the past %05.3f' % float(avg_steps)
            pos = n // state['checkFreq']
            state['rho'] = float(rho)
            state['Omega'] = float(avg_reg)
            state['train_nll'] = float(avg_cost)
            state['valid_error'] =  float(error)
            state['gradient_norm'] = float(avg_norm)
            store_valid[pos] = error
            store_rho[pos] = rho
            if time.time() - last_save > state['saveFreq']*60:
                if channel is not None:
                    channel.save()
                numpy.savez(state['name']+'_state.npz',
                            train_nll = store_train,
                            valid_error = store_valid,
                            gradient_norm = store_norm,
                            rho_Whh = store_rho,
                            Omega = store_reg,
                            W_uh = W_uh.get_value(),
                            W_hh = W_hh.get_value(),
                            W_hy = W_hy.get_value(),
                            b_hh = b_hh.get_value(),
                            b_hy = b_hy.get_value())
                last_save = time.time()

            if error < best_score:
                best_score = error
                state['bestvalid_nll'] = float(valid_cost)
                state['bestvalid_error'] = float(error)
            if error < .0001 and numpy.isfinite(valid_cost):
                cont = False
                print '**> Iter %07d'%n,':',\
                    'train nnl %05.3f' % avg_cost,\
                    'valid error %07.3f%%'% error,\
                    'best valid error %07.3f%%'% best_score,\
                    'average gradient norm %6.3f'% avg_norm, \
                    'rho_Whh %5.2f'% rho, \
                    'Omega %5.2f'% float(avg_reg), \
                    'alpha %6.3f' % alpha, \
                    'steps in the past %05.3f' % float(avg_steps)
                solved=1
                print '!!!!! STOPING - Problem solved'
            avg_cost = 0
            avg_norm = 0
            avg_reg = 0
            avg_steps = 0
            avg_len = 0
            avg_time = 0
        state['steps'] = n

    state['steps'] = n
    if solved:
        state['solved'] = 1
    else:
        state['solved'] = 0
    if channel is not None:
        channel.save()
    numpy.savez(state['name']+'_final_state.npz',
                train_nll = store_train,
                valid_error = store_valid,
                gradient_norm = store_norm,
                rho_Whh = store_rho,
                Omega = store_reg,
                W_uh = W_uh.get_value(),
                W_hh = W_hh.get_value(),
                W_hy = W_hy.get_value(),
                b_hh = b_hh.get_value(),
                b_hy = b_hy.get_value())


if __name__=='__main__':
    # Define hyperparameters
    #
    # note: this code is meant to work with Jobman
    # (http://deeplearning.net/software/jobman), though it runs without it
    # as well
    state = {}
    # Number of hidden units
    state['nhid'] = 50
    # Random seed
    state['seed'] = 52
    # Task to execute. Pick from:
    #   * torder  - temporal order task
    #   * torder3 - 3 bit temporal order task
    #   * add     - addition task
    #   * mul     - multiplication task
    #   * mem     - memorization task
    #   * perm    - random permutation task
    state['task'] = 'torder3'
    # Pick network initialization style. It has to be one of the 3 variants
    # described in the paper, i.e.:
    #   * sigmoid
    #   * basic_tanh
    #   * smart_tanh
    state['init'] = 'smart_tanh'
    # Strength of the regularization term proposed in the paper
    state['alpha'] = 2.
    # Learning rate
    state['lr'] = .01
    # Maximal length of the task and minimal length of the task.
    # If you want to run an experiment were sequences have fixed length, set
    # these to hyper-parameters to the same value. Otherwise each batch will
    # have a length randomly sampled from [min_length, max_length]
    state['max_length'] = 200
    state['min_length'] = 50
    # batch size
    state['bs'] = 20
    # Size of the batch over which the evaluation error is computed
    state['ebs'] = 10000
    # Computational batch size used during evaluation. This means that these
    # many samples will be evaluated in parallel at a time during validation
    # phase. Set this value according to the amount of memory available on
    # your machine
    state['cbs'] = 1000
    # How often do we compute test error
    state['checkFreq'] = 2000
    # Constant used for numerical stability. When computing the
    # regularizationt term, values for which dC/dx_k is smaller than `bound`
    # are not considered
    state['bound'] =1e-20
    # If we should do gradient clipping or not. Please set to `rescale` if
    # you want to do gradient clipping, otherwise to `nothing`
    state['clipstyle'] = 'rescale'
    # Threshold for gradient clipping, if clipstyle is set to rescale
    state['cutoff'] = 1.
    # Maximal number of iterations
    state['maxiters'] = int(5e6)
    # How often to save to disk current state of the model, in minutes
    state['saveFreq'] = 5
    # Prefix to be appended to name of the file in which the state of the
    # experiemt is stored.
    state['name'] = 'test'
    jobman(state, None)
