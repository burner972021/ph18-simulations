import random
from math import sqrt, cos, sin, pi
from quantum_library import QuantumState, QuantumOperation

# The unitary matrices of Alice and Bob's possible operations.
U_alice_0 = [[1, 0], [0, 1]]
U_alice_1 = [[1/sqrt(2), -1/sqrt(2)], [1/sqrt(2), 1/sqrt(2)]]
U_bob_0 = [[cos(pi/8), -sin(pi/8)], [sin(pi/8), cos(pi/8)]]
U_bob_1 = [[cos(pi/8), sin(pi/8)], [-sin(pi/8), cos(pi/8)]]

# Alice and Bob win when their input (a, b)
# and their response (s, t) satisfy this relationship.
def win(a, b, s, t):
    return (a and b) == (s != t)

wins = 0

for i in range(10000):
    # Alice and Bob share an entangled state
    state = QuantumState([1/sqrt(2), 0, 0, 1/sqrt(2)])

    # The input to alice and bob is random
    a = random.choice([True, False])
    b = random.choice([True, False])

    # Alice chooses her operation based on her input
    if a == 0:
        alice_op = QuantumOperation(U_alice_0)
    if a == 1:
        alice_op = QuantumOperation(U_alice_1)

    # Bob chooses his operation based on his input
    if b == 0:
        bob_op = QuantumOperation(U_bob_0)
    if b == 1:
        bob_op = QuantumOperation(U_bob_1)

    # We combine Alice and Bob's operations
    combined_operation = alice_op.compose(bob_op)    

    # Alice and Bob make their measurements
    result = combined_operation.apply(state).measure()

    # Convert the 4 state measurement result to two 1-bit results
    if result == 0:
        s, t = False, False
    if result == 1:
        s, t = False, True
    if result == 2:
        s, t = True, False
    if result == 3:
        s, t = True, True

    # Check if they won and add it to the total
    wins += win(a, b, s, t)

print('They won this many times:', wins)