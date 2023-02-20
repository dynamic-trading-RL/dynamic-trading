
import os

if __name__ == '__main__':

    print(__file__)
    print(os.getcwd())

    folders_2b_emptied = (
        '/resources/data/data_tmp/',
        '/resources/data/financial_time_series_data/financial_time_series_calibrations/',
        '/resources/data/financial_time_series_data/financial_time_series_info/',
        '/resources/data/supervised_regressors/',
        '/resources/figures/backtesting/',
        '/resources/figures/residuals/',
        '/resources/figures/on_the_fly_backtesting/',
        '/resources/figures/on_the_fly_simulationtesting/',
        '/resources/figures/polynomial/',
        '/resources/figures/simulationtesting/',
        '/resources/figures/time_series/',
        '/resources/figures/training/',
        '/resources/reports/calibrations/',
        '/resources/reports/backtesting/',
        '/resources/reports/model_choice/',
        '/resources/reports/simulationtesting/',
        '/resources/reports/training/'
    )

    for folder in folders_2b_emptied:
        folderpath = os.path.dirname(__file__) + folder
        for filename in os.listdir(folderpath):
            if filename == '.keep':
                continue
            filepath = folderpath + filename
            os.remove(filepath)
