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

class AddTask(object):
    def __init__(self, rng, floatX):
        self.rng = rng
        self.floatX = floatX
        self.nin = 2
        self.nout = 1
        self.classifType='lastLinear'

    def generate(self, batchsize, length):
        l = self.rng.randint(int(length*.1))+length
        p0 = self.rng.randint(int(l*.1), size=(batchsize,))
        p1 = self.rng.randint(int(l*.4), size=(batchsize,)) + int(l*.1)
        data = self.rng.uniform(size=(l, batchsize, 2)).astype(self.floatX)
        data[:,:,0] = 0.
        data[p0, numpy.arange(batchsize), numpy.zeros((batchsize,),
                                                      dtype='int32')] = 1.
        data[p1, numpy.arange(batchsize), numpy.zeros((batchsize,),
                                                      dtype='int32')] = 1.

        targs = (data[p0, numpy.arange(batchsize),
                     numpy.ones((batchsize,), dtype='int32')] + \
                 data[p1, numpy.arange(batchsize),
                      numpy.ones((batchsize,), dtype='int32')])/2.
        return data, targs.reshape((-1,1)).astype(self.floatX)


if __name__ == '__main__':
    print 'Testing add task generator ..'
    addtask = AddTask(numpy.random.RandomState(123), 'float32')
    seq, targ = addtask.generate(3, 25)
    assert seq.dtype == 'float32'
    assert targ.dtype == 'float32'
    print 'Seq_0'
    print seq[:,0,:]
    print 'Targ0', targ[0]
    print
    print 'Seq_1'
    print seq[:,1,:]
    print 'Targ1', targ[1]
    print
    print 'Seq_2'
    print seq[:,2,:]
    print 'Targ2', targ[2]
