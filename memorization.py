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

class MemTask(object):
    def __init__(self,
                  rng,
                  floatX,
                  n_values = 5,
                  n_pos = 10,
                  generate_all = False):
        self.rng = rng
        self.floatX = floatX
        self.dim = n_values**n_pos
        self.n_values = n_values
        self.n_pos = n_pos
        self.generate_all = generate_all
        if generate_all:
            self.data = numpy.zeros((n_pos, self.dim, n_values+2))
            for val in xrange(self.dim):
                tmp_val = val
                for k in xrange(n_pos):
                    self.data[k, val, tmp_val % n_values] = 1.
                    tmp_val = tmp_val // n_values
        self.nin = self.n_values + 2
        self.nout = n_values + 1
        self.classifType = 'softmax'
        self.report = 'all'


    def generate(self, batchsize, length):

        if self.generate_all:
            batchsize = self.dim
        input_data = numpy.zeros((length + 2*self.n_pos,
                                  batchsize,
                                  self.n_values + 2),
                                 dtype=self.floatX)
        targ_data = numpy.zeros((length + 2*self.n_pos,
                                 batchsize,
                                 self.n_values+1),
                                dtype=self.floatX)
        targ_data[:-self.n_pos,:, -1] = 1
        input_data[self.n_pos:,:, -2] = 1
        input_data[length + self.n_pos, :, -2] = 0
        input_data[length + self.n_pos, :, -1] = 1

        if not self.generate_all:
            self.data = numpy.zeros((self.n_pos, batchsize, self.n_values+2))
            for val in xrange(batchsize):
                tmp_val = self.rng.randint(self.dim)
                for k in xrange(self.n_pos):
                    self.data[k, val, tmp_val % self.n_values] = 1.
                    tmp_val = tmp_val // self.n_values
        input_data[:self.n_pos, :, :] = self.data
        targ_data[-self.n_pos:, :, :] = self.data[:,:,:-1]
        return input_data, targ_data.reshape(((length +
                                               2*self.n_pos)*batchsize, -1))

if __name__ == '__main__':
    print 'Testing memorization task generator ..'
    task = MemTask(numpy.random.RandomState(123),
                   'float32')
    seq, targ = task.generate(3, 25)
    assert seq.dtype == 'float32'
    assert targ.dtype == 'float32'
    print 'Seq_0'
    print seq[:,0,:].argmax(axis=1)
    print 'Targ0'
    print targ.reshape((25+2*10, 3, -1))[:,0,:].argmax(1)
    print
    print 'Seq_1'
    print seq[:,1,:].argmax(axis=1)
    print 'Targ1'
    print targ.reshape((25+2*10, 3, -1))[:,1,:].argmax(1)
    print
    print 'Seq_2'
    print seq[:,2,:].argmax(axis=1)
    print 'Targ2'
    print targ.reshape((25+2*10, 3, -1))[:,2,:].argmax(1)
