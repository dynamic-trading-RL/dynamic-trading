
import os

if __name__ == '__main__':

    print(__file__)
    print(os.getcwd())

    folders_2b_emptied = (
        '/data/data_tmp/',
        '/data/financial_time_series_data/financial_time_series_calibrations/',
        '/data/financial_time_series_data/financial_time_series_info/',
        '/data/supervised_regressors/',
        '/figures/backtesting/',
        '/figures/on_the_fly_backtesting/',
        '/figures/on_the_fly_simulationtesting/',
        '/figures/polynomial/',
        '/figures/simulationtesting/',
        '/figures/training/',
        '/reports/calibrations/',
        '/reports/sharpe_ratios/',
        '/reports/t-tests/'
    )

    for folder in folders_2b_emptied:
        folderpath = os.path.dirname(__file__) + folder
        for filename in os.listdir(folderpath):
            if filename == '.keep':
                continue
            filepath = folderpath + filename
            os.remove(filepath)
            # try:
            #     filepath = folderpath + filename
            #     os.remove(filepath)
            # except:
            #     continue
