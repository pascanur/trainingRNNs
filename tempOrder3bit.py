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

class TempOrder3bitTask(object):
     def __init__(self, rng, floatX):
        self.rng = rng
        self.floatX = floatX
        self.nin = 6
        self.nout = 8
        self.classifType='lastSoftmax'

     def generate(self, batchsize, length):
        l = length
        p0 = self.rng.randint(int(l*.1), size=(batchsize,)) + int(l*.1)
        v0 = self.rng.randint(2, size=(batchsize,))
        p1 = self.rng.randint(int(l*.1), size=(batchsize,)) + int(l*.3)
        v1 = self.rng.randint(2, size=(batchsize,))
        p2 = self.rng.randint(int(l*.1), size=(batchsize,)) + int(l*.6)
        v2 = self.rng.randint(2, size=(batchsize,))
        targ_vals = v0 + v1*2 + v2 * 4
        vals  = self.rng.randint(4, size=(l, batchsize))+2
        vals[p0, numpy.arange(batchsize)] = v0
        vals[p1, numpy.arange(batchsize)] = v1
        vals[p2, numpy.arange(batchsize)] = v2
        data = numpy.zeros((l, batchsize, 6), dtype=self.floatX)
        targ = numpy.zeros((batchsize, 8), dtype=self.floatX)
        data.reshape((l*batchsize, 6))[numpy.arange(l*batchsize),
                                    vals.flatten()] = 1.
        targ[numpy.arange(batchsize), targ_vals] = 1.
        return data, targ


if __name__ == '__main__':
    print 'Testing temp Order task generator ..'
    task = TempOrder3bitTask(numpy.random.RandomState(123), 'float32')
    seq, targ = task.generate(3, 25)
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
