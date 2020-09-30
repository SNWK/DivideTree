import numpy as np

def iscross(line1, line2):
        A, B = line1
        C, D = line2
        AC = C - A
        AD = D - A
        BC = C - B
        BD = D - B
        CA = - AC
        CB = - BC
        DA = - AD
        DB = - BD
        
        return np.cross(AC,AD)*np.cross(BC,BD) < 0 and np.cross(CA,CB)*np.cross(DA,DB) < 0

line1 = [np.array([0, 0]), np.array([1, 1])]
line2 = [np.array([0, 1]), np.array([1, 1])]

print(iscross(line1, line2))