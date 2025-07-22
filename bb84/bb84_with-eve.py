import random

# Number of qubits to simulate
N = 3000

def generate_bits(n):
    return [random.randint(0, 1) for _ in range(n)]

def generate_bases(n):
    return [random.choice(['Z', 'X']) for _ in range(n)] 

def measure(qubit_bit, qubit_basis, measure_basis):
    if qubit_basis == measure_basis:
        return qubit_bit 
    else:
        return random.randint(0, 1)
    
# Simulation
alice_bits = generate_bits(N)
alice_bases = generate_bases(N)

eve_bases = generate_bases(N)
eve_bits = []

# Eve intercepts and resends qubits to Bob
intercepted_bits = []
intercepted_bases = []

for i in range(N):
    eve_bit = measure(alice_bits[i], alice_bases[i], eve_bases[i])
    eve_bits.append(eve_bit)
    intercepted_bits.append(eve_bit)
    intercepted_bases.append(eve_bases[i])

bob_bases = generate_bases(N)
bob_bits = []

for i in range(N):
    bob_bit = measure(intercepted_bits[i], intercepted_bases[i], bob_bases[i])
    bob_bits.append(bob_bit)

# Sifting step: Alice and Bob compare their bases
sifted_key_indices = [i for i in range(N) if alice_bases[i] == bob_bases[i]]

# Form sifted keys using indices where bases matched
alice_key = [alice_bits[i] for i in sifted_key_indices]
bob_key = [bob_bits[i] for i in sifted_key_indices]

# Estimate error rate (Eve's interference)
sample_size = min(20, len(alice_key) // 2)  # Check 20 bits or half the key
sample_indices = random.sample(range(len(alice_key)), sample_size)

errors = 0
for idx in sample_indices:
    if alice_key[idx] != bob_key[idx]:
        errors += 1

error_rate = errors / sample_size

print(f"Sifted key length: {len(alice_key)} bits")
print(f"Tested {sample_size} bits - Found {errors} errors")
print(f"Estimated error rate: {error_rate:.2%}")

# Security threshold (25% is theoretical maximum for BB84 with Eve)
if error_rate > 0.0001:
    print("Eve detected! Aborting key exchange.")
else:
    # Remove tested bits from final key
    final_key = [bit for i, bit in enumerate(alice_key) if i not in sample_indices]
    print(f"Secure key established: {len(final_key)} bits")