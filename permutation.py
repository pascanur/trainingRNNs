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
import numpy

class PermTask(object):
    def __init__(self, rng, floatX):
        self.rng = rng
        self.floatX = floatX
        self.nin = 100
        self.nout = 100
        self.classifType = 'lastSoftmax'
        self.report = 'last'

    def generate(self, batchsize, length):
        randvals = self.rng.randint(98, size=(length+1, batchsize)) + 2
        val = self.rng.randint(2, size=(batchsize,))
        randvals[numpy.zeros((batchsize,), dtype='int32'),
                 numpy.arange(batchsize)] = val
        randvals[numpy.ones((batchsize,), dtype='int32')*length,
                 numpy.arange(batchsize)] = val
        _targ = randvals[1:]
        _inp = randvals[:-1]
        inp = numpy.zeros((length, batchsize, 100), dtype=self.floatX)
        # targ = numpy.zeros((length, batchsize, 100), dtype=self.floatX)
        targ = numpy.zeros((1, batchsize, 100), dtype=self.floatX)
        inp.reshape((length*batchsize, 100))[\
                numpy.arange(length*batchsize),
                _inp.flatten()] = 1.
        #targ.reshape((length*batchsize, 100))[\
        #        numpy.arange(batchsize),
        #        _targ[-1].flatten()] = 1.
        targ.reshape((batchsize, 100))[\
                numpy.arange(batchsize),
                _targ[-1].flatten()] = 1.
        return inp, targ.reshape((batchsize, 100))


if __name__ == '__main__':
    print 'Testing permutation task generator ..'
    task = PermTask(numpy.random.RandomState(123), 'float32')
    seq, targ = task.generate(3, 25)
    assert seq.dtype == 'float32'
    assert targ.dtype == 'float32'
    print 'Seq_0'
    print seq[:,0,:].argmax(axis=1)
    print 'Targ0'
    print targ[0].argmax(axis=0)
    print
    print 'Seq_1'
    print seq[:,1,:].argmax(axis=1)
    print 'Targ1'
    print targ[1].argmax(axis=0)
    print
    print 'Seq_2'
    print seq[:,2,:].argmax(axis=1)
    print 'Targ2'
    print targ[2].argmax(axis=0)
