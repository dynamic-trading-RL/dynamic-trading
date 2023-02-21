import os
import numpy as np
from scipy import stats


class TTester:
    """
    Class for performing a t-test. Refer to :obj:`scipy.stats.ttest_ind` for more details.

    Attributes
    ----------
    alternative : {'two-sided', 'less', 'greater'}
        Defines the alternative hypothesis. The following options are available (default is 'two-sided'):
        - 'two-sided': the means of the distributions underlying the samples are unequal.
        - 'less': the mean of the distribution underlying the first sample is less than the mean of the distribution
        underlying the second sample.
        - 'greater': the mean of the distribution underlying the first sample is greater than the mean of the
        distribution underlying the second sample.
    equal_var : bool
        If True (default), perform a standard independent 2 sample test that assumes equal population variances. If
        False, perform Welch's t-test, which does not assume equal population variance.
    nan_policy : {'propagate', 'raise', 'omit'}
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
        - 'propagate': returns nan
        - 'raise': throws an error
        - 'omit': performs the calculations ignoring nan values
    permutations : non-negative int, np.inf, or None (default)
        If 0 or None (default), use the t-distribution to calculate p-values. Otherwise, `permutations` is  the number
        of random permutations that will be used to estimate p-values using a permutation test. If `permutations` equals
        or exceeds the number of distinct partitions of the pooled data, an exact test is performed instead (i.e. each
        distinct partition is used exactly once). See Notes for details.
    random_state : {None, int, `numpy.random.Generator`}
        If `seed` is None the `numpy.random.Generator` singleton is used. If `seed` is an int, a new ``Generator``
        instance is used, seeded with `seed`. If `seed` is already a ``Generator`` instance then that instance is used.
        Pseudorandom number generator state used for generating random permutations.
    sample_a, sample_b : array_like
        The samples being tested.
    t_test_id : str
        An ID for this t-test.
    t_test_result : dict
        Dictionary containing the results of the t-test.

    """

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
        """
        Execute t-test.

        """

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
        """
        Print t-test result in output folder /resources/reports/simulationtesting/.

        """

        out_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/resources/reports/simulationtesting/'
        file_name = f'{self._on_the_fly_str}{self._n_str}t_test_result_{self.t_test_id}.txt'

        with open(out_path + file_name, 'w') as f:
            for key, value in self.t_test_result.items():
                f.write(f'{key}:{value}\n')


if __name__ == '__main__':

    n = 1000

    a = np.random.randn(n)
    b = np.random.randn(n)

    tTester = TTester(t_test_id='test', sample_a=a, sample_b=b)
