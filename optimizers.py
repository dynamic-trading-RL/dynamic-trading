class Optimizers:

    def __init__(self):
        self._shgo = 0
        self._dual_annealing = 0
        self._differential_evolution = 0
        self._brute = 0
        self._local = 0

    def __repr__(self):
        return 'Used optimizers:\n' + \
               '  shgo: ' + str(self._shgo) + '\n' + \
               '  dual_annealing: ' + str(self._dual_annealing) + '\n' + \
               '  differential_evolution: ' + str(self._differential_evolution) + '\n' + \
               '  brute: ' + str(self._brute) + '\n' + \
               '  local: ' + str(self._local)

    def __str__(self):
        return 'Used optimizers:\n' + \
               '  shgo: ' + str(self._shgo) + '\n' + \
               '  dual_annealing: ' + str(self._dual_annealing) + '\n' + \
               '  differential_evolution: ' + str(self._differential_evolution) + '\n' + \
               '  brute: ' + str(self._brute) + '\n' + \
               '  local: ' + str(self._local)
