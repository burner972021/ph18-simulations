# https://medium.com/@alexandre.laplante/a-quantum-computing-library-in-48-lines-of-python-bfa242fb9acb

import numpy
import random

def is_unitary(M):
    M_star = numpy.transpose(M).conjugate()
    identity = numpy.eye(len(M))
    return numpy.allclose(identity, numpy.matmul(M_star, M))

class QuantumState:
    def __init__(self, vector):
        length = numpy.linalg.norm(vector)
        if not abs(1 - length) < 0.00001:
            raise ValueError('Quantum states must be unit length.')
        self.vector = numpy.array(vector)

    def measure(self):
        choices = range(len(self.vector))
        weights = [abs(a)**2 for a in self.vector]
        outcome = random.choices(choices, weights)[0]

        new_state = numpy.zeros(len(self.vector))
        new_state[outcome] = 1
        self.vector = new_state
        return outcome 

    def compose(self, state):
        new_vector = numpy.kron(self.vector, state.vector)
        return QuantumState(new_vector)

    def __repr__(self):
        return '<QuantumState: {}>'.format(', '.join(map(str, self.vector)))

class QuantumOperation:
    def __init__(self, matrix):
        if not is_unitary(matrix):
            raise ValueError('Quantum operations must be unitary')
        self.matrix = matrix

    def apply(self, state):
        new_vector = numpy.matmul(self.matrix, state.vector)
        return QuantumState(new_vector)

    def compose(self, operation):
        new_matrix = numpy.kron(self.matrix, operation.matrix)
        return QuantumOperation(new_matrix)

    def __repr__(self):
        return '<QuantumOperation: {}>'.format(str(self.matrix))