import numpy as np
import random

n = 1000

alice_angles = [-22.5, 0.0, 22.5, 45.0]
bob_angles = [-22.5, 0.0, 22.5, 45.0]

s1_counts = [[0] * 4] * 4
s2_counts = [[0] * 4] * 4
store1 = {'10': 0, '12': 0, '32': 0, '30': 0}
store2 = {'01': 0, '21': 0, '23': 0, '03': 0}

def calc4b():


for _ in range(n):
    a = random.randint(0, 3)
    b = random.randint(0, 3)
    