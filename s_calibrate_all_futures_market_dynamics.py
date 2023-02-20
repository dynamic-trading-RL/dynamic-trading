from src.market_utils.calibrator import AllSeriesDynamicsCalibrator

if __name__ == '__main__':

    allSeriesDynamicsCalibrator = AllSeriesDynamicsCalibrator()
    allSeriesDynamicsCalibrator.fit_all_series_dynamics()
    allSeriesDynamicsCalibrator.print_all_series_dynamics_results()

    print('--- End s_calibrate_all_futures_market_dynamics.py')
