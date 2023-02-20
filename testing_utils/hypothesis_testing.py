import os
import numpy as np
from scipy import stats


class TTester:

    def __init__(self,
                 t_test_id: str,
                 sample_a: np.ndarray,
                 sample_b: np.ndarray,
                 equal_var: bool = False,
                 nan_policy: str = 'omit',
                 permutations=None,
                 random_state: int = 789,
                 alternative: str = 'two-sided',
                 on_the_fly: bool = False,
                 n: int = None):

        self.t_test_id = t_test_id
        self.sample_a = sample_a
        self.sample_b = sample_b
        self.equal_var = equal_var
        self.nan_policy = nan_policy
        self.permutations = permutations
        self.random_state = random_state
        self.alternative = alternative
        self._on_the_fly = on_the_fly
        if self._on_the_fly:
            self._on_the_fly_str = 'on_the_fly_'
        else:
            self._on_the_fly_str = ''

        self._n = n
        if self._n is not None:
            self._n_str = f'{self._n}_'
        else:
            self._n_str = ''

        self.t_test_result = {'statistic': np.nan,
                              'pvalue': np.nan}

        self.execute_t_test()
        self.print_t_test_result()

    def execute_t_test(self):

        statistic, pvalue = stats.ttest_ind(a=self.sample_a,
                                            b=self.sample_b,
                                            equal_var=self.equal_var,
                                            nan_policy=self.nan_policy,
                                            permutations=self.permutations,
                                            random_state=self.random_state,
                                            alternative=self.alternative)
        self.t_test_result['statistic'] = statistic
        self.t_test_result['pvalue'] = pvalue

    def print_t_test_result(self):

        out_path = os.path.dirname(os.path.dirname(__file__)) + '/reports/simulationtesting/'
        file_name = f'{self._on_the_fly_str}{self._n_str}t_test_result_{self.t_test_id}.txt'

        with open(out_path + file_name, 'w') as f:
            for key, value in self.t_test_result.items():
                f.write(f'{key}:{value}\n')


if __name__ == '__main__':

    n = 1000

    a = np.random.randn(n)
    b = np.random.randn(n)

    tTester = TTester(t_test_id='test', sample_a=a, sample_b=b)
