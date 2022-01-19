import numpy as np
import sys


class Block:
    def __init__(self, N, XY, R):
        # Check values
        if (N < 1) | (N > 10000):
            raise Exception('Dimensions should be within 1 to 10000')

        if (XY[0] < 1) | (XY[1] > N):
            raise Exception('Pizzeria location must be within {} x {}'.format(N, N))

        if (R < 1) | (R > 100):
            raise Exception('Pizzeria range cannot be zero or exceed 100')
        self.N = N
        self.XY = XY
        self.R = R
        self.block = np.zeros((N, N))

    def build_block(self):
        # Function to build a 2 dimensional array showing the range of pizzeria. If it's within range set to 1
        # and outside range set to 0
        x = self.XY[0] - 1
        y = self.XY[1] - 1

        for i in range(self.R + 1):
            if (x + i) < N:
                self.block[x + i][y] = 1
            if (x - i) >= 0:
                self.block[x - i][y] = 1
            if (y + i) < N:
                self.block[x][y + i] = 1
            if (y - i) >= 0:
                self.block[x][y - i] = 1

        for i in range(1, self.R):
            if (x + i < N) & (y + i < N):
                self.block[x + i][y + i] = 1
            if (x - i >= 0) & (y - i >= 0):
                self.block[x - i][y - i] = 1
            if (x + i < N) & (y - i >= 0):
                self.block[x + i][y - i] = 1
            if (x - i >= 0) & (y + i < N):
                self.block[x - i][y + i] = 1

        self.block = np.flipud(self.block)

        return self.block


def data(file_name):
    datz = sys.stdin.read().split('\n')
    f_datz = datz[0].split()

    N = int(f_datz[0])
    M = int(f_datz[1])

    pizz = []
    for i in range(1, M + 1):
        dat = [int(l) for l in datz[i].split()]
        pizz.append(dat)

    return N, M, pizz


if __name__ == '__main__':
    file_name = 'pizzeria_input.txt'

    N, M, pizzeria = data(file_name)

    sum_range = np.zeros((N, N))
    for i in pizzeria:
        block = Block(N, i[0:-1], i[-1])
        pizz_range = block.build_block()
        sum_range += pizz_range

    anz = int(np.max(sum_range))
    sys.stdout.write(str(anz))
