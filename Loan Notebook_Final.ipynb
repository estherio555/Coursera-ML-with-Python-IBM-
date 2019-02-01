{
    "nbformat_minor": 2, 
    "cells": [
        {
            "source": "<a href=\"https://www.bigdatauniversity.com\"><img src=\"https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png\" width=\"400\" align=\"center\"></a>\n\n<h1 align=\"center\"><font size=\"5\">Classification with Python</font></h1>", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "source": "In this notebook we try to practice all the classification algorithms that we learned in this course.\n\nWe load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.\n\nLets first load required libraries:", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 3, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [], 
            "source": "import itertools\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom matplotlib.ticker import NullFormatter\nimport pandas as pd\nimport numpy as np\nimport matplotlib.ticker as ticker\nfrom sklearn import preprocessing\n%matplotlib inline"
        }, 
        {
            "source": "### About dataset", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "source": "This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:\n\n| Field          | Description                                                                           |\n|----------------|---------------------------------------------------------------------------------------|\n| Loan_status    | Whether a loan is paid off on in collection                                           |\n| Principal      | Basic principal loan amount at the                                                    |\n| Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |\n| Effective_date | When the loan got originated and took effects                                         |\n| Due_date       | Since it\u2019s one-time payoff schedule, each loan has one single due date                |\n| Age            | Age of applicant                                                                      |\n| Education      | Education of applicant                                                                |\n| Gender         | The gender of applicant                                                               |", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "source": "Lets download the dataset", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 4, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "--2019-02-01 02:39:45--  https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv\nResolving s3-api.us-geo.objectstorage.softlayer.net (s3-api.us-geo.objectstorage.softlayer.net)... 67.228.254.193\nConnecting to s3-api.us-geo.objectstorage.softlayer.net (s3-api.us-geo.objectstorage.softlayer.net)|67.228.254.193|:443... connected.\nHTTP request sent, awaiting response... 200 OK\nLength: 23101 (23K) [text/csv]\nSaving to: \u2018loan_train.csv\u2019\n\n100%[======================================>] 23,101      --.-K/s   in 0.002s  \n\n2019-02-01 02:39:45 (14.6 MB/s) - \u2018loan_train.csv\u2019 saved [23101/23101]\n\n"
                }
            ], 
            "source": "!wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv"
        }, 
        {
            "source": "### Load Data From CSV File  ", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 5, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "execution_count": 5, 
                    "metadata": {}, 
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Unnamed: 0</th>\n      <th>Unnamed: 0.1</th>\n      <th>loan_status</th>\n      <th>Principal</th>\n      <th>terms</th>\n      <th>effective_date</th>\n      <th>due_date</th>\n      <th>age</th>\n      <th>education</th>\n      <th>Gender</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>0</td>\n      <td>0</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>9/8/2016</td>\n      <td>10/7/2016</td>\n      <td>45</td>\n      <td>High School or Below</td>\n      <td>male</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>2</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>9/8/2016</td>\n      <td>10/7/2016</td>\n      <td>33</td>\n      <td>Bechalor</td>\n      <td>female</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>3</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>15</td>\n      <td>9/8/2016</td>\n      <td>9/22/2016</td>\n      <td>27</td>\n      <td>college</td>\n      <td>male</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>4</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>9/9/2016</td>\n      <td>10/8/2016</td>\n      <td>28</td>\n      <td>college</td>\n      <td>female</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>6</td>\n      <td>6</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>9/9/2016</td>\n      <td>10/8/2016</td>\n      <td>29</td>\n      <td>college</td>\n      <td>male</td>\n    </tr>\n  </tbody>\n</table>\n</div>", 
                        "text/plain": "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n0           0             0     PAIDOFF       1000     30       9/8/2016   \n1           2             2     PAIDOFF       1000     30       9/8/2016   \n2           3             3     PAIDOFF       1000     15       9/8/2016   \n3           4             4     PAIDOFF       1000     30       9/9/2016   \n4           6             6     PAIDOFF       1000     30       9/9/2016   \n\n    due_date  age             education  Gender  \n0  10/7/2016   45  High School or Below    male  \n1  10/7/2016   33              Bechalor  female  \n2  9/22/2016   27               college    male  \n3  10/8/2016   28               college  female  \n4  10/8/2016   29               college    male  "
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "df = pd.read_csv('loan_train.csv')\ndf.head()"
        }, 
        {
            "execution_count": 6, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "execution_count": 6, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "(346, 10)"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "df.shape"
        }, 
        {
            "source": "### Convert to date time object ", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 7, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "execution_count": 7, 
                    "metadata": {}, 
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Unnamed: 0</th>\n      <th>Unnamed: 0.1</th>\n      <th>loan_status</th>\n      <th>Principal</th>\n      <th>terms</th>\n      <th>effective_date</th>\n      <th>due_date</th>\n      <th>age</th>\n      <th>education</th>\n      <th>Gender</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>0</td>\n      <td>0</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>2016-09-08</td>\n      <td>2016-10-07</td>\n      <td>45</td>\n      <td>High School or Below</td>\n      <td>male</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>2</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>2016-09-08</td>\n      <td>2016-10-07</td>\n      <td>33</td>\n      <td>Bechalor</td>\n      <td>female</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>3</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>15</td>\n      <td>2016-09-08</td>\n      <td>2016-09-22</td>\n      <td>27</td>\n      <td>college</td>\n      <td>male</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>4</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>2016-09-09</td>\n      <td>2016-10-08</td>\n      <td>28</td>\n      <td>college</td>\n      <td>female</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>6</td>\n      <td>6</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>2016-09-09</td>\n      <td>2016-10-08</td>\n      <td>29</td>\n      <td>college</td>\n      <td>male</td>\n    </tr>\n  </tbody>\n</table>\n</div>", 
                        "text/plain": "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n0           0             0     PAIDOFF       1000     30     2016-09-08   \n1           2             2     PAIDOFF       1000     30     2016-09-08   \n2           3             3     PAIDOFF       1000     15     2016-09-08   \n3           4             4     PAIDOFF       1000     30     2016-09-09   \n4           6             6     PAIDOFF       1000     30     2016-09-09   \n\n    due_date  age             education  Gender  \n0 2016-10-07   45  High School or Below    male  \n1 2016-10-07   33              Bechalor  female  \n2 2016-09-22   27               college    male  \n3 2016-10-08   28               college  female  \n4 2016-10-08   29               college    male  "
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "df['due_date'] = pd.to_datetime(df['due_date'])\ndf['effective_date'] = pd.to_datetime(df['effective_date'])\ndf.head()"
        }, 
        {
            "source": "# Data visualization and pre-processing\n\n", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "source": "Let\u2019s see how many of each class is in our data set ", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 8, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "execution_count": 8, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "PAIDOFF       260\nCOLLECTION     86\nName: loan_status, dtype: int64"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "df['loan_status'].value_counts()"
        }, 
        {
            "source": "260 people have paid off the loan on time while 86 have gone into collection \n", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "source": "Lets plot some columns to underestand data better:", 
            "cell_type": "markdown", 
            "metadata": {}
        }, 
        {
            "execution_count": 27, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "Solving environment: ...working... failed\n"
                }, 
                {
                    "output_type": "stream", 
                    "name": "stderr", 
                    "text": "\nCondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://repo.continuum.io/pkgs/r/win-64/repodata.json.bz2>\nElapsed: -\n\nAn HTTP error occurred when trying to retrieve this URL.\nHTTP errors are often intermittent, and a simple retry will get you on your way.\nSSLError(MaxRetryError('HTTPSConnectionPool(host=\\'repo.continuum.io\\', port=443): Max retries exceeded with url: /pkgs/r/win-64/repodata.json.bz2 (Caused by SSLError(SSLError(\"bad handshake: Error([(\\'SSL routines\\', \\'ssl3_get_server_certificate\\', \\'certificate verify failed\\')],)\",),))',),)\n\n\n"
                }
            ], 
            "source": "# notice: installing seaborn might takes a few minutes\n#!conda install -c anaconda seaborn -y"
        }, 
        {
            "execution_count": 9, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "display_data", 
                    "data": {
                        "image/png": "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IIcccgjnnHMODzzwQMt2ffr0aXkkxwUXXMD8+fP53Oc+11L3nDlzGDFiBMOGDePwww9nzpw5LesuvPBCBg4cyMaNGwFYu3YtNTU1HfqdtJcft2FmZW3lO29z1HWPFK2+Z7+Vb+zsrUUEZ555JhMnTmwZtbyhoYHVq1dz4YUXMnXqVMaNG8f69es566yzuPPOO1tGiuiIL3zhCy2jnDfbsGEDp5xyCt/97nc59dRTAXjiiSeorq5uGX5p9OjRTJkypWW8vvnz57fs/+KLL3LFFVcwb948amtreeONNzjxxBM54IADGDFiBJA8W2r69OlccsklHY65I3wFZWZWZE888QRVVVVcfPHFLWV1dXUsX76cY489lnHjxgGwyy67cPvtt3PTTTcV7dj33XcfRx99dEtyAvjsZz/b8kDEtkyZMoVrr72W2tpaAGpra7nmmmu49dZbW7aZPHkyt912G5s3by5a3Pk4QZmZFdnSpUs/8UgMyP+ojCFDhvDhhx/ywQcfdPg4ud12dXV1fPTRR60eu73a8ziPwYMHc9xxx/HjH/+408dpD3fxmZl1k20fi5GrtfLtydfFV6h8MeYru/baaznttNM45ZRTinr8XL6CMjMrskMPPZRFixblLd92XMXXX3+dPn360Ldv3y49dkf23zbGfI/zOPDAA6mrq+PBBx/s9LHa4gRlZlZkY8aMYePGjdx9990tZc8//zxDhw5lwYIF/PrXvwaSBxtefvnlXHnllUU79he/+EV+85vf8POf/7yl7JFHHuGll15q1/5XXHEF3/72t2lsbASgsbGRG2+8ka9//euf2PYb3/gGU6ZMKUrc+biLz8zK2oCB+7XrzruO1NcWScyePZvJkydz00030bt3b2pqavje977HQw89xFe/+lUuvfRStmzZwvnnn7/VreUzZszY6rbuZ55JngE7YsQIdtghuaY455xzGDFiBA888MBWz5O68847OeaYY5g7dy6TJ09m8uTJVFVVMWLECL7//e+36/XV1dVx8803c+qpp7Jp0yaqqqq45ZZbWp4QnOvQQw9l1KhR7X50fUe163EbXc2PE7Du4MdtVAY/biM7Cn3cRptdfJKmS3pP0tKcslslvSppiaTZknZPy2skfSSpIZ1+2N5AzMzMcrXnM6gZwLbXx/OAwyJiBLAcuCZn3e8joi6dLsbMzKwT2kxQEfEk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\n", 
                        "text/plain": "<matplotlib.figure.Figure at 0x7f406c903f98>"
                    }, 
                    "metadata": {}
                }
            ], 
            "source": "import seaborn as sns\n\nbins = np.linspace(df.Principal.min(), df.Principal.max(), 10)\ng = sns.FacetGrid(df, col=\"Gender\", hue=\"loan_status\", palette=\"Set1\", col_wrap=2)\ng.map(plt.hist, 'Principal', bins=bins, ec=\"k\")\n\ng.axes[-1].legend()\nplt.show()"
        }, 
        {
            "execution_count": 13, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "output_type": "display_data", 
                    "data": {
                        "image/png": "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\n", 
                        "text/plain": "<matplotlib.figure.Figure at 0x7f7019d05828>"
                    }, 
                    "metadata": {}
                }
            ], 
            "source": "bins = np.linspace(df.age.min(), df.age.max(), 10)\ng = sns.FacetGrid(df, col=\"Gender\", hue=\"loan_status\", palette=\"Set1\", col_wrap=2)\ng.map(plt.hist, 'age', bins=bins, ec=\"k\")\n\ng.axes[-1].legend()\nplt.show()"
        }, 
        {
            "source": "# Pre-processing:  Feature selection/extraction", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "source": "### Lets look at the day of the week people get the loan ", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 10, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "output_type": "display_data", 
                    "data": {
                        "image/png": "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\n", 
                        "text/plain": "<matplotlib.figure.Figure at 0x7f4064113cc0>"
                    }, 
                    "metadata": {}
                }
            ], 
            "source": "df['dayofweek'] = df['effective_date'].dt.dayofweek\nbins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)\ng = sns.FacetGrid(df, col=\"Gender\", hue=\"loan_status\", palette=\"Set1\", col_wrap=2)\ng.map(plt.hist, 'dayofweek', bins=bins, ec=\"k\")\ng.axes[-1].legend()\nplt.show()\n"
        }, 
        {
            "source": "We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 ", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 11, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "execution_count": 11, 
                    "metadata": {}, 
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Unnamed: 0</th>\n      <th>Unnamed: 0.1</th>\n      <th>loan_status</th>\n      <th>Principal</th>\n      <th>terms</th>\n      <th>effective_date</th>\n      <th>due_date</th>\n      <th>age</th>\n      <th>education</th>\n      <th>Gender</th>\n      <th>dayofweek</th>\n      <th>weekend</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>0</td>\n      <td>0</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>2016-09-08</td>\n      <td>2016-10-07</td>\n      <td>45</td>\n      <td>High School or Below</td>\n      <td>male</td>\n      <td>3</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>2</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>2016-09-08</td>\n      <td>2016-10-07</td>\n      <td>33</td>\n      <td>Bechalor</td>\n      <td>female</td>\n      <td>3</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>3</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>15</td>\n      <td>2016-09-08</td>\n      <td>2016-09-22</td>\n      <td>27</td>\n      <td>college</td>\n      <td>male</td>\n      <td>3</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>4</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>2016-09-09</td>\n      <td>2016-10-08</td>\n      <td>28</td>\n      <td>college</td>\n      <td>female</td>\n      <td>4</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>6</td>\n      <td>6</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>2016-09-09</td>\n      <td>2016-10-08</td>\n      <td>29</td>\n      <td>college</td>\n      <td>male</td>\n      <td>4</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n</div>", 
                        "text/plain": "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n0           0             0     PAIDOFF       1000     30     2016-09-08   \n1           2             2     PAIDOFF       1000     30     2016-09-08   \n2           3             3     PAIDOFF       1000     15     2016-09-08   \n3           4             4     PAIDOFF       1000     30     2016-09-09   \n4           6             6     PAIDOFF       1000     30     2016-09-09   \n\n    due_date  age             education  Gender  dayofweek  weekend  \n0 2016-10-07   45  High School or Below    male          3        0  \n1 2016-10-07   33              Bechalor  female          3        0  \n2 2016-09-22   27               college    male          3        0  \n3 2016-10-08   28               college  female          4        1  \n4 2016-10-08   29               college    male          4        1  "
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)\ndf.head()"
        }, 
        {
            "source": "## Convert Categorical features to numerical values", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "source": "Lets look at gender:", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 12, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "execution_count": 12, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "Gender  loan_status\nfemale  PAIDOFF        0.865385\n        COLLECTION     0.134615\nmale    PAIDOFF        0.731293\n        COLLECTION     0.268707\nName: loan_status, dtype: float64"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)"
        }, 
        {
            "source": "86 % of female pay there loans while only 73 % of males pay there loan\n", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "source": "Lets convert male to 0 and female to 1:\n", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 20, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [], 
            "source": "df['Gender'].replace(to_replace=['male', 'female'], value=[0 ,1],inplace=True)"
        }, 
        {
            "source": "## One Hot Encoding  \n#### How about education?", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 21, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "execution_count": 21, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "education             loan_status\nBechalor              PAIDOFF        0.750000\n                      COLLECTION     0.250000\nHigh School or Below  PAIDOFF        0.741722\n                      COLLECTION     0.258278\nMaster or Above       COLLECTION     0.500000\n                      PAIDOFF        0.500000\ncollege               PAIDOFF        0.765101\n                      COLLECTION     0.234899\nName: loan_status, dtype: float64"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "df.groupby(['education'])['loan_status'].value_counts(normalize=True)"
        }, 
        {
            "source": "#### Feature befor One Hot Encoding", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 22, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "execution_count": 22, 
                    "metadata": {}, 
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Principal</th>\n      <th>terms</th>\n      <th>age</th>\n      <th>Gender</th>\n      <th>education</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>45</td>\n      <td>0</td>\n      <td>High School or Below</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>33</td>\n      <td>1</td>\n      <td>Bechalor</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>1000</td>\n      <td>15</td>\n      <td>27</td>\n      <td>0</td>\n      <td>college</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>28</td>\n      <td>1</td>\n      <td>college</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>29</td>\n      <td>0</td>\n      <td>college</td>\n    </tr>\n  </tbody>\n</table>\n</div>", 
                        "text/plain": "   Principal  terms  age  Gender             education\n0       1000     30   45       0  High School or Below\n1       1000     30   33       1              Bechalor\n2       1000     15   27       0               college\n3       1000     30   28       1               college\n4       1000     30   29       0               college"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "df[['Principal','terms','age','Gender','education']].head()"
        }, 
        {
            "source": "#### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame ", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 23, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "execution_count": 23, 
                    "metadata": {}, 
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Principal</th>\n      <th>terms</th>\n      <th>age</th>\n      <th>Gender</th>\n      <th>weekend</th>\n      <th>Bechalor</th>\n      <th>High School or Below</th>\n      <th>college</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>45</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>33</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>1000</td>\n      <td>15</td>\n      <td>27</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>28</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>29</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n</div>", 
                        "text/plain": "   Principal  terms  age  Gender  weekend  Bechalor  High School or Below  \\\n0       1000     30   45       0        0         0                     1   \n1       1000     30   33       1        0         1                     0   \n2       1000     15   27       0        0         0                     0   \n3       1000     30   28       1        1         0                     0   \n4       1000     30   29       0        1         0                     0   \n\n   college  \n0        0  \n1        0  \n2        1  \n3        1  \n4        1  "
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "Feature = df[['Principal','terms','age','Gender','weekend']]\nFeature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)\nFeature.drop(['Master or Above'], axis = 1,inplace=True)\nFeature.head()\n"
        }, 
        {
            "source": "### Feature selection", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "source": "Lets defind feature sets, X:", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 24, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "execution_count": 24, 
                    "metadata": {}, 
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Principal</th>\n      <th>terms</th>\n      <th>age</th>\n      <th>Gender</th>\n      <th>weekend</th>\n      <th>Bechalor</th>\n      <th>High School or Below</th>\n      <th>college</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>45</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>33</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>1000</td>\n      <td>15</td>\n      <td>27</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>28</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>29</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n</div>", 
                        "text/plain": "   Principal  terms  age  Gender  weekend  Bechalor  High School or Below  \\\n0       1000     30   45       0        0         0                     1   \n1       1000     30   33       1        0         1                     0   \n2       1000     15   27       0        0         0                     0   \n3       1000     30   28       1        1         0                     0   \n4       1000     30   29       0        1         0                     0   \n\n   college  \n0        0  \n1        0  \n2        1  \n3        1  \n4        1  "
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "X = Feature\nX[0:5]"
        }, 
        {
            "source": "What are our lables?", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 25, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "execution_count": 25, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "array(['PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF'], dtype=object)"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "y = df['loan_status'].values\ny[0:5]"
        }, 
        {
            "source": "## Normalize Data ", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "source": "Data Standardization give data zero mean and unit variance (technically should be done after train test split )", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 26, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "execution_count": 26, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "array([[ 0.51578458,  0.92071769,  2.33152555, -0.42056004, -1.20577805,\n        -0.38170062,  1.13639374, -0.86968108],\n       [ 0.51578458,  0.92071769,  0.34170148,  2.37778177, -1.20577805,\n         2.61985426, -0.87997669, -0.86968108],\n       [ 0.51578458, -0.95911111, -0.65321055, -0.42056004, -1.20577805,\n        -0.38170062, -0.87997669,  1.14984679],\n       [ 0.51578458,  0.92071769, -0.48739188,  2.37778177,  0.82934003,\n        -0.38170062, -0.87997669,  1.14984679],\n       [ 0.51578458,  0.92071769, -0.3215732 , -0.42056004,  0.82934003,\n        -0.38170062, -0.87997669,  1.14984679]])"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "X= preprocessing.StandardScaler().fit(X).transform(X)\nX[0:5]"
        }, 
        {
            "source": "# Classification ", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "source": "Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model\nYou should use the following algorithm:\n- K Nearest Neighbor(KNN)\n- Decision Tree\n- Support Vector Machine\n- Logistic Regression\n\n\n\n__ Notice:__ \n- You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.\n- You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.\n- You should include the code of the algorithm in the following cells.", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "source": "# K Nearest Neighbor(KNN)\nNotice: You should find the best k to build the model with the best accuracy.  \n**warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.", 
            "cell_type": "markdown", 
            "metadata": {}
        }, 
        {
            "execution_count": 38, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "(276, 8) (70, 8) (276,) (70,)\n\n[[ 0.51578458  0.92071769 -0.65321055 -0.42056004 -1.20577805 -0.38170062\n   1.13639374 -0.86968108]\n [-1.31458942 -0.95911111  2.16570687  2.37778177  0.82934003 -0.38170062\n   1.13639374 -0.86968108]\n [ 0.51578458  0.92071769 -0.15575453 -0.42056004 -1.20577805 -0.38170062\n  -0.87997669  1.14984679]\n [-1.31458942 -0.95911111 -0.3215732  -0.42056004  0.82934003 -0.38170062\n  -0.87997669  1.14984679]\n [ 0.51578458  0.92071769 -0.81902922  2.37778177  0.82934003 -0.38170062\n  -0.87997669  1.14984679]]\n"
                }
            ], 
            "source": "from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.metrics import classification_report, confusion_matrix\n\n#Splitting data into train and test set\nX_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=0)\nprint(X_train.shape, X_test.shape, y_train.shape, y_test.shape)\nprint()\nprint(X_train[0:5])"
        }, 
        {
            "execution_count": 41, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "execution_count": 41, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "array([ 0.65714286,  0.57142857,  0.7       ,  0.67142857,  0.71428571,\n        0.68571429,  0.75714286,  0.72857143,  0.75714286,  0.68571429,\n        0.77142857,  0.77142857,  0.8       ,  0.75714286])"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "k=15\nmean_acc = np.zeros((k-1))\nstd_acc=np.zeros((k-1))\n\nfor n in range(1,k):\n    #Train Model and Predict \n    neig = KNeighborsClassifier(n_neighbors =n).fit(X_train, y_train)\n    yhat = neig.predict(X_test)\n    mean_acc[n-1]=np.mean(yhat==y_test);\n    \n    #std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])\n    \nmean_acc\n"
        }, 
        {
            "execution_count": 42, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "execution_count": 42, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "[<matplotlib.lines.Line2D at 0x7f4062d4ecf8>]"
                    }, 
                    "output_type": "execute_result"
                }, 
                {
                    "output_type": "display_data", 
                    "data": {
                        "image/png": "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\n", 
                        "text/plain": "<matplotlib.figure.Figure at 0x7f4062db37f0>"
                    }, 
                    "metadata": {}
                }
            ], 
            "source": "#Plot model accuracy for Different number of Neighbors\n\nplt.plot(range(1,k),mean_acc,'g')           #)           , mean_train, 'b')"
        }, 
        {
            "execution_count": 48, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [], 
            "source": "#print( \"The best accuracy was with\", mean_acc.max(), \"with k=\", mean_acc.argmax()+1) "
        }, 
        {
            "execution_count": 52, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "             precision    recall  f1-score   support\n\n COLLECTION       0.33      0.42      0.37        12\n    PAIDOFF       0.87      0.83      0.85        58\n\navg / total       0.78      0.76      0.77        70\n\n"
                }
            ], 
            "source": "#print(confusion_matrix(y_test,yhat))\n\nprint(classification_report(y_test,yhat))"
        }, 
        {
            "execution_count": 34, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "0.757142857143\n0.735507246377\n"
                }
            ], 
            "source": "yhat_prob = neig.predict_prob(X_test)\nmean_test = accuracy_score(y_test, yhat)\nmean_train = accuracy_score(y_train, neig.predict(X_train))\nprint(mean_test)\nprint(mean_train)"
        }, 
        {
            "source": "# Decision Tree", 
            "cell_type": "markdown", 
            "metadata": {}
        }, 
        {
            "execution_count": 55, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [], 
            "source": "from sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.metrics import classification_report, confusion_matrix"
        }, 
        {
            "execution_count": 65, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "execution_count": 65, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "array([ 0.82857143,  0.82857143,  0.82857143,  0.71428571,  0.8       ,\n        0.71428571,  0.72857143,  0.71428571,  0.71428571])"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "k=10\nmean_dec = np.zeros((k-1))\n\nfor n in range(1,k):\n    #Train Model and Predict \n    dectree = DecisionTreeClassifier(criterion=\"entropy\", max_depth = n)\n    dectree.fit(X_train, y_train)\n    yhat_dec = dectree.predict(X_test)\n    mean_dec[n-1] = np.mean(yhat_dec==y_test)\n\nmean_dec"
        }, 
        {
            "execution_count": 66, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "execution_count": 66, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "[<matplotlib.lines.Line2D at 0x7f40621ddc50>]"
                    }, 
                    "output_type": "execute_result"
                }, 
                {
                    "output_type": "display_data", 
                    "data": {
                        "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX0AAAD8CAYAAACb4nSYAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4wLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvpW3flQAAIABJREFUeJzt3Xl4XOWV5/Hv0e5dXmSVvEglsxjLm64wxsaritABkkAWOsHTCU0eaNIsSScTJiHpdCZDmjSZZIZ0GGchkEDotN0OgYQkBBK65AVjg1d5kW1sLNmSbcnybtnadeYPVbllWbJKVlXdWs7nefSgqrp175Gxf3V13ve+V1QVY4wxySHF7QKMMcZEj4W+McYkEQt9Y4xJIhb6xhiTRCz0jTEmiVjoG2NMErHQN8aYJGKhb4wxScRC3xhjkkia2wV0N2bMGPV6vW6XYYwxcWXTpk3HVDWnr+1iLvS9Xi8bN250uwxjjIkrInIglO2svWOMMUnEQt8YY5KIhb4xxiQRC31jjEkiFvrGGJNELPSNMSaJWOgbY0wSibl5+lfqXMs5vrv2u26XEVdSJIV7i+/Fm+11uxRjTJQkTOifbz3PP6/+Z7fLiCuKcqLxBD+87Ydul2KMiZKECf2cITl0/M8Ot8uIK/N/Pp/NRza7XYYxJoqsp5/EHI9DeV05HWoflsYkCwv9JFaSV0JDSwP7TuxzuxRjTJRY6CcxJ88BsBaPMUnEQj+JFeUUkZ6SzpYjW9wuxRgTJRb6SSwjNYNpY6expdZC35hkYaGf5ErySthSuwVVdbsUY0wUWOgnOcfjcOz8MWrO1LhdijEmCiz0k1xwMNdaPMYkBwv9JDczdyaC2GCuMUkipNAXkVtFZI+I7BORx3p4PV9EykRki4hsE5HbA8/fIiKbRGR74L++cP8AZmCGZAxh8pjJdqZvTJLoM/RFJBVYCtwGFAFLRKSo22bfAFaoqgPcDfwo8Pwx4COqOh34W+DFcBVuwsfxODZX35gkEcqZ/mxgn6ruV9UWYDlwZ7dtFBge+H4EcBhAVbeo6uHA8zuBLBHJHHjZJpwcj0P1mWqOnz/udinGmAgLJfTHA9VdHtcEnuvqW8CnRaQGeA34fA/7+QSwRVWbr6BOE0EleSWADeYakwxCCX3p4bnuk7qXAM+r6gTgduBFEbmwbxGZCnwX+FyPBxB5QEQ2isjG+vr60Co3YWPLMRiTPEIJ/RpgYpfHEwi0b7q4D1gBoKrrgCxgDICITABeAe5R1fd7OoCqPqOqs1R1Vk5OTv9+AjNgowaNIn9Evp3pG5MEQgn9DcA1IlIoIhl0DtS+2m2bg8DNACIyhc7QrxeRbOCPwNdUdW34yjbh5ngcm7ZpTBLoM/RVtQ14BHgD2EXnLJ2dIvK4iNwR2OzLwN+JSDmwDLhXO6/rfwS4GvgnEdka+BobkZ/EDEhJXgnvHX+PhpYGt0sxxkRQSHfOUtXX6Byg7frcN7t8XwHM6+F9/wzYPQzjgONxUJTy2nLm5V/yv9IYkyDsilwD2HIMxiQLC30DwPhh48kZnGN9fWMSnIW+AUBEcPIcO9M3JsFZ6JsLHI/DjqM7aGlvcbsUY0yEWOibCxyPQ2tHKzuP7nS7FGNMhFjomwtsOQZjEp+FvrngqlFXMSxjmC3HYEwCs9A3F6RICjM9M+1M35gEZqFvLuJ4HMpry2nvaHe7FGNMBFjom4uU5JVwrvUc+07sc7sUY0wEWOibizgeW2bZmERmoW8uUpRTREZqhvX1jUlQFvrmIump6UwbO81C35gEZaFvLlHiKWHLkS10ro5tjEkkFvrmEk6ew/HG41Sfqe57Y2NMXLHQN5cIDubaipvGJB4LfXOJGbkzSJEU6+sbk4As9M0lhmQMYfLoyTZt05gEZKFvemRr6xuTmCz0TY8cj0PNmRqOnT/mdinGmDCy0Dc9urDMsg3mGpNQLPRNj4o9xYAtx2BMorHQNz0aNWgUBSMKrK9vTIKx0De9ssFcYxKPhb7pVYmnhL3H93K2+azbpRhjwsRC3/TKyXNQlPK6crdLMcaESUihLyK3isgeEdknIo/18Hq+iJSJyBYR2SYit3d57WuB9+0RkQ+Gs3gTWbYcgzGJJ62vDUQkFVgK3ALUABtE5FVVreiy2TeAFar6YxEpAl4DvIHv7wamAuOAN0XkWlW1e/HFgXHDxpEzOMf6+sYkkFDO9GcD+1R1v6q2AMuBO7tto8DwwPcjgMOB7+8Elqtqs6pWAvsC+zNxQEQoySuxaZsBHdphy02buBdK6I8Huq6xWxN4rqtvAZ8WkRo6z/I/34/3mhjmeBx21u+kua3Z7VJcpaoU/6SYb/i/4XYpxgxIKKEvPTzX/XRnCfC8qk4AbgdeFJGUEN+LiDwgIhtFZGN9fX0IJZlocfIc2jra2Fm/0+1SXPX+yffZfnQ7L+9+2e1SjBmQUEK/BpjY5fEE/qt9E3QfsAJAVdcBWcCYEN+Lqj6jqrNUdVZOTk7o1ZuIs+UYOvkr/QDsPrabw2cv+StsTNwIJfQ3ANeISKGIZNA5MPtqt20OAjcDiMgUOkO/PrDd3SKSKSKFwDXAu+Eq3kTepJGTGJYxLOn7+mVVZaSnpAOwsmqlu8UYMwB9hr6qtgGPAG8Au+icpbNTRB4XkTsCm30Z+DsRKQeWAfdqp510/gZQAbwOPGwzd+JLiqRQ7ClO6hk8qoq/0s9dRXeRnZV94azfmHjU55RNAFV9jc4B2q7PfbPL9xXAvF7e+wTwxABqNC5zPA7PbnmW9o52UlNS3S4n6irqKzh67igfmPQBGtsaLfRNXLMrck2fSvJKON96nr0n9rpdiivKqsoA8BX68Hl9VJ6qpOpUlbtFGXOFLPRNn5y8zitzk7Wv76/048324s32UlpYCkBZZZnLVRlzZSz0TZ+mjJlCZmpmUs7g6dAOVlatxOf1ATA1Zyo5g3PwV1mLx8QnC33Tp/TUdKaNnZaUg7nlteWcbDqJr7Az9EUEX6GPssoyuzrXxCULfROS4HIMyRZ0wUHbYFsHoNRbyqGzh5J2jMPENwt9ExLH43Cy6SQHTx90u5So8lf5mTx6MuOGjbvwXPCs32bxmHhkoW9CEhzMTaYWT2t7K6sPrL4Q8kFXj7qaCcMnWOibuGShb0IyI3cGKZKSVIO5m45soqGl4ZLQFxFKvaWsrFpJh3a4VJ0xV8ZC34RkcPpgrhtzHZtrk2faZvBMfrF38SWv+Qp91J+vZ+fR5F6IzsQfC30TMsfjJNWZvr/Sz4zcGYwZPOaS10q9pRe2MSaeWOibkDkeh0NnD1F/LvGXv25ua2Zt9doL8/O7K8gu4KqRV124WteYeGGhb0J2YZnlJBjMXV+znqa2poumanYX7Ou3d9gagiZ+WOibkBV7ioHkWI7BX+knRVJYWLCw1218hT5ON59Oig9Bkzgs9E3IRg4aiTfbmxQhV1ZVxvV515Odld3rNrYOj4lHFvqmX5JhMPdcyznW16y/MFjbG89QD1PGTLF1eExcsdA3/VKSV8LeE3s503zG7VIiZm31Wlo7Wi+Zn98TX6GPNQfW0NreGoXKjBk4C33TL46n88rc8tpylyuJHH+ln7SUNObnz+9zW1+hj3Ot59hweEMUKjNm4Cz0Tb8kw3IMZVVlzJkwhyEZQ/rcdlHBIgSx+fombljom37JG5rH2CFjEzb0TzedZuPhjX3284NGDx7NTM9MC30TNyz0Tb+IyIVllhPR6gOr6dCOkPr5QT6vj7er36aprSmClRkTHhb6pt8cj0NFfQXNbc1ulxJ2ZVVlZKVlMWfCnJDfU1pYSnN7M+uq10WwMmPCw0Lf9JvjcWjraGPH0R1ulxJ2/ko/N028iay0rJDfs7BgIamSai0eExcs9E2/Jepg7rHzxyivK+91vZ3eDM8czqxxs2y+vokLFvqm3yaNnMTwzOEJ19dfVbUKoF/9/CBfoY93D71LQ0tDuMsyJqws9E2/pUgKxZ7ihDvT91f6GZI+hFnjZvX7vaXeUto62njr4FsRqMyY8LHQN1fE8TiU15Yn1AqT/io/CwsWkp6a3u/3zsufR3pKuvX1TcwLKfRF5FYR2SMi+0TksR5ef0pEtga+3hORU11e+98islNEdonID0VEwvkDGHeU5JXQ2NbInuN73C4lLI6cPcLuY7uvqLUDnXcWmztxrq2vb2Jen6EvIqnAUuA2oAhYIiJFXbdR1S+parGqFgNPAy8H3nsTMA+YAUwDbgAWhfUnMK4ILseQKIuvBcM61IuyelLqLWXzkc2cbDwZrrKMCbtQzvRnA/tUdb+qtgDLgTsvs/0SYFngewWygAwgE0gH6q68XBMrrhtzHZmpmQnT1/dX+snOyr5wz4Ar4Sv00aEdrD6wOoyVGRNeoYT+eKC6y+OawHOXEJECoBDwA6jqOqAMOBL4ekNVd/XwvgdEZKOIbKyvT/xb8SWC9NR0pudOT6jQX+xdTGpK6hXv48bxNzIobZC1eExMCyX0e+rBay/b3g28pKrtACJyNTAFmEDnB4VPRC65FZGqPqOqs1R1Vk5OTmiVG9eVeDqXY1Dt7a9DfKg6VUXlqcp+z8/vLjMtk3n582ww18S0UEK/BpjY5fEE4HAv297Nf7V2AD4GrFfVBlVtAP4EhH59u4lpTp7DqaZTHDh9wO1SBiR456vL3Q83VD6vj+1Ht3P03NEB78uYSAgl9DcA14hIoYhk0Bnsr3bfSEQmAyOBrguQHAQWiUiaiKTTOYh7SXvHxKdEGcz1V/nJGZzD1JypA95XcPbPyqqVA96XMZHQZ+irahvwCPAGnYG9QlV3isjjInJHl02XAMv14t/1XwLeB7YD5UC5qv4+bNUbV03PnU6KpMR1X19VKassw1foIxyzia8fdz3DMobZfXNNzEoLZSNVfQ14rdtz3+z2+Fs9vK8d+NwA6jMxbHD6YKaMmRLXyzHsPbGXQ2cPDWiqZldpKWksLFho6/CYmGVX5JoBcfKcuD7TDw66XulFWT3xFfp47/h7HDpzKGz7NCZcLPTNgDgeh8NnD1PXEJ+XX5RVlTFh+ASuHnV12PYZ/ACxqZsmFlnomwEpySsB4nOZ5Q7toKyyjFJvaVj6+UEzcmcwatAom7ppYpKFvhmQ4BWs8TiDZ+fRndSfrw9rawc6VyFd7F1soW9ikoW+GZDsrGwKswvj8kw/GMrhGsTtyuf1ceD0ASpPVoZ938YMhIW+GbB4HcwtqyrjqpFXUZBdEPZ9By/0srN9E2ss9M2AlXhK2HdiH6ebTrtdSsjaO9pZWbUyImf5AFPGTCF3SK5N3TQxx0LfDFjwnrnldeUuVxK6LbVbON18Ouz9/CARwVfow1/pj/u1iUxisdA3AxaPyzGEc72d3pR6S6ltqE2YG82YxGChbwYsb1geuUNy46qv76/yM2XMFDxDPRE7RvC3COvrm1hioW/CoiSvJG6WY2htb2XNgTURa+0ETRo5ifwR+Rb6JqZY6JuwcDwOFfUVNLU1uV1KnzYc3sC51nMRD30RodRbysqqlXRoR0SPZUyoLPRNWDh5Du3azo6jO9wupU/+Sj+CsKgg8rdr9hX6ON54nO112yN+LGNCYaFvwiI4mBsPLR5/pZ+ZnpmMHjw64scKTgm1Fo+JFRb6JiwmjZzEiMwRMT+Dp6mtiber3x7wrRFDNXHERK4ZdY0tvmZihoW+CQsRodhTHPMzeNZVr6O5vTni/fyuSr2lrDqwiraOtqgd05jeWOibsHE8DtvqtsV0uPkr/aRKKgsKFkTtmL5CH2eaz8RF68skPgt9EzYleSU0tjWy51jsXozkr/Iza9wshmcOj9oxF3sXA9gtFE1MsNA3YRNcjiFWWzwNLQ28e+jdqLZ2AHKH5jI1Z6qtw2NigoW+CZvrxlxHVlpWzA7mvnXwLdo62iK2yNrl+Ap9rDmwhpb2lqgf25iuLPRN2KSlpDF97PSYPdP3V/pJT0lnXv68qB/bV+ijsa2Rd2reifqxjenKQt+EVUleCVtqt8TkypJlVWXMnTiXwemDo37sRQWLEMSmbhrXWeibsHI8DqeaTlF1qsrtUi5ysvEkm49sdqW1AzBy0EicPMcu0jKus9A3YRWrg7mrD6ymQzuiPojblc/rY13NOhpbG12rwRgLfRNW08dOJ1VSY24wt6yqjEFpg7hx/I2u1VBaWEpLewtvV7/tWg3GhBT6InKriOwRkX0i8lgPrz8lIlsDX++JyKkur+WLyJ9FZJeIVIiIN3zlm1gzKH0QU3KmsLk2ti5E8lf6mZ8/n8y0TNdqWJC/gFRJtRaPcVVaXxuISCqwFLgFqAE2iMirqloR3EZVv9Rl+88DTpdd/BJ4QlX/IiJDAVtjNsE5Hoc397/pdhkXHD13lO1Ht7Nk2hJX6xiWOYzZ42fbfH3jqlDO9GcD+1R1v6q2AMuBOy+z/RJgGYCIFAFpqvoXAFVtUNXzA6zZxDjH43Ck4Qi1DbVulwLAyqqVAK7284NKvaVsOLSBs81n3S7FJKlQQn88UN3lcU3guUuISAFQCARPZa4FTonIyyKyRUS+F/jNwSSwC4O5MdLXL6ssY1jGMK4fd73bpeAr9NGu7aw5uMbtUkySCiX0pYfnepuEfTfwkqq2Bx6nAQuAR4EbgEnAvZccQOQBEdkoIhvr6+tDKMnEsmJPMRA7M3j8VX4WFiwkLaXPbmbE3TTxJjJSM6yvb1wTSujXABO7PJ4AHO5l27sJtHa6vHdLoDXUBvwWKOn+JlV9RlVnqeqsnJyc0Co3MSs7K5tJIyfFROgfOnOI946/FxOtHegc6L5p4k0W+sY1oYT+BuAaESkUkQw6g/3V7huJyGRgJLCu23tHikgwyX1ARff3msTjeJyYaO8Er4CNldCHzr7+1tqtnGg84XYpJgn1GfqBM/RHgDeAXcAKVd0pIo+LyB1dNl0CLNcu198H2jyPAv8pItvpbBX9LJw/gIlNJXklvH/yfU43nXa1Dn+ln1GDRjEjd4ardXTlK/ShKKuqVrldiklCITU5VfU14LVuz32z2+Nv9fLevwCx8y/OREXwnrlba7eyyBv5G5D3xl/pZ7F3MSkSO9chzh4/m8Hpg/FX+vnYlI+5XY5JMrHzL8EklFhYjqHyZCUHTh+I2v1wQ5WRmsH8/Pm2+JpxhYW+iQjPUA+eoR5XQz84WBpL/fwgn9fHzvqd1DXUuV2KSTIW+iZiSvJKXL0vrL/Kj2eoh+vGXOdaDb0JfhDZ2b6JNgt9EzGOx2FX/S5XVpVUVfyVfkq9pYj0dKmJu5w8h+GZw+2+uSbqLPRNxDgeh3ZtZ8fRHVE/9p7je6htqI3J1g503mVsUcEiW4fHRJ2FvomY4GCuGy2eYD/frZumhMJX6GPfiX1Un67ue2NjwsRC30RMYXYhIzJHuDKY66/0kz8in0kjJ0X92KEKfiBZX99Ek4W+iRgRwclzoh76HdrByqqV+Ap9MdnPD5qeO53Rg0bbkgwmqiz0TUQ5Hodtddto62iL2jG3123neOPxmG7tAKRICqWFpfgr/TF5I3mTmCz0TUQ5HoemtiZ2H9sdtWPGQz8/yOf1UX2mmvdPvu92KSZJWOibiCrJ61xUNZqLr5VVlXHNqGuYOGJi3xu7rLQw0Ne3qZsmSiz0TURNHjOZrLSsqPX12zraWHVgVcxO1exu8ujJ5A3Ns6mbJmos9E1EpaWkMSN3RtRCf/ORzZxpPhMXrR3oHOz2Ffooqyyzvr6JCgt9E3ElnhK2HNkSlVALtkkWexdH/FjhUuotpe5cHbuO7XK7FJMELPRNxDl5DqebT1N5qjLix/JX+Zk2dhq5Q3MjfqxwCbaibOqmiQYLfRNxwbX1Iz2Y29LewpoDa+KmtRNUOLIQb7bXQt9EhYW+ibjpudNJldSIL8fwTs07NLY1xs0gblel3lJWVq2kQzvcLsUkOAt9E3FZaVkU5RRFfDC3rKoMQVhU4N6duq6Ur9DHyaaTlNeWu12KSXAW+iYqorEcg7/Sj5PnMHLQyIgeJxKCLSlr8ZhIs9A3UeF4HGobajly9khE9t/Y2si6mnUxd2vEUI0fPp5rR19ri6+ZiLPQN1FxYTA3Qmf7b1e/TUt7S1z284N8Xh+rDqyitb3V7VJMArPQN1FR7CkGIjeDx1/pJy0ljfn58yOy/2jwFfpoaGlg05FNbpdiEpiFvomKEVkjuGrkVRE70/dX+blh3A0MyxwWkf1HQ/CCMuvrm0iy0DdRE6nB3LPNZ9lwaENct3YAcobkMH3sdOvrm4iy0DdR43gc9p/cz6mmU2Hd75qDa2jX9rgPfehs8bx18C2a25rdLsUkKAt9EzXBZZa31m4N6379lX4yUjOYO2FuWPfrBl+hj6a2JtbXrHe7FJOgQgp9EblVRPaIyD4ReayH158Ska2Br/dE5FS314eLyCER+X/hKtzEn0gtx+Cv9HPTxJsYlD4orPt1w8KChaRIirV4TMT0GfoikgosBW4DioAlIlLUdRtV/ZKqFqtqMfA08HK33XwbWBWekk28yh2aS97QPDbXhm85hhONJ9hauzVu5+d3l52VTUleiQ3mmogJ5Ux/NrBPVferaguwHLjzMtsvAZYFH4jI9UAu8OeBFGoSQ0leSVjP9FdVrULRC3egSgQ+r4/1Nes533re7VJMAgol9McD1V0e1wSeu4SIFACFgD/wOAX4P8D/uNwBROQBEdkoIhvr6+tDqdvEKcfjsPvYbhpbG8OyP3+ln8Hpg5k9fnZY9hcLSgtLae1oZe3BtW6XYhJQKKEvPTzX290w7gZeUtX2wOOHgNdUtbqX7Tt3pvqMqs5S1Vk5OTkhlGTilZPn0K7tbD+6PSz7K6sqY0H+AjJSM8Kyv1gwP38+aSlp1uIxERFK6NcAXe8wPQE43Mu2d9OltQPMBR4RkSrg+8A9IvLkFdRpEkRwMDccyyzXNdSxs35n3K2f35ehGUO5cfyNdt9cExGhhP4G4BoRKRSRDDqD/dXuG4nIZGAksC74nKr+jarmq6oXeBT4papeMvvHJA9vtpfsrOyw9PWDM1wSYX5+d6XeUjYe3sjpptNul2ISTJ+hr6ptwCPAG8AuYIWq7hSRx0Xkji6bLgGWq93d2VyGiOB4wnNlblllGSMyR+DkOWGoLLb4Cn10aAdrDq5xuxSTYEKap6+qr6nqtap6lao+EXjum6r6apdtvnW5s3hVfV5VHxl4ySbeOR6HbXXbBryapL/KzyLvItJS0sJUWeyYO3EumamZ1tc3YWdX5Jqoc/Icmtub2X1s9xXvo/p0NftO7Eu4fn5QVloW8/LnWeibsLPQN1EXXI5hIC2eRO7nB5V6SymvK+f4+eNul2ISiIW+ibrJoyczKG3QgAZz/ZV+xgwew7Sx08JYWWwJfqCtrFrpbiEmoVjom6hLTUllRu6MK16OQVXxV/pZ7F1MiiTuX+Ebxt3AkPQh1uIxYZW4/2JMTCvJK2Fr7VY6tKPf733/5PtUn6lOmPV2epOems6CggW2+JoJKwt94wrH43Cm+QyVJyv7/d6yysTv5wf5vD52HdsVsRvKm+RjoW9cEZxbfyWDuf4qP3lD87h29LXhLivmBD/Y7GzfhIuFvnHFtLHTSJXUfi/HoKqUVZbhK/Qh0tOyUIml2FNMdlb2hd9ujBkoC33jiqy0LKaOndrvM/1dx3ZRd64uKVo70Dnovahgka3DY8LGQt+4xvE4/Z62GZzJkiyhD50/6/6T+6k6VeV2KSYBWOgb1zgeh7pzdf0apPRX+vFme/FmeyNXWIwJXnVsLR4TDhb6xjXBwdxQ+/od2sHKqpUJP1Wzu6ljp5IzOMcGc01YWOgb1xR7ioHQZ/CU15ZzsulkUrV2AFIkhdLCUvyVfmwRWzNQFvrGNcMzh3P1qKtDDv1gPz+R7ocbKp/Xx6Gzh9h7Yq/bpYSkQzt4sfxFXt1zya03jMss9I2rHI8TcnvHX+Vn8ujJjBs2LsJVxZ7gB1089PX3Ht+L7wUf9/z2Hu5cfiefWPEJu7gshljoG1c5HoeqU1WcbDx52e1a21tZfWB10rV2gq4ZdQ3jh42P6ambbR1tfG/t95jxkxlsrd3Kzz7yM568+Un++N4fKfpREc9vfd7aUzHAQt+4KrjM8tbarZfdbtORTTS0NCTs+vl9ERF8hT7KKstiMji31W1j7nNz+cqbX+GDV32QiocruL/kfr46/6tse3Ab08dO57O/+ywf/LcP2tRTl1noG1eFuhxDsJ+/2Ls40iXFrFJvKfXn69lZv9PtUi5obmvmm2Xf5Ppnrufg6YOsuGsFr3zqlYtacNeOvpaV965k6e1LWVezjmk/msbT7zx9RYvtmYGz0DeuGjtkLOOGjeuzr19WVcaM3BnkDMmJUmWxJ9jaipWlltdVr8P5qcO3V3+bJdOWUPFQBX899a97XB4jRVJ46IaH2PnQThYULOALr3+BBb9YwK76XS5Untws9I3rSvJKLnum39zWzFsH30q6+fndFWQXMGnkJNdD/1zLOb74+heZ9/N5NLQ08Np/e41ffuyXjB48us/35o/I79z+o79k97HdFP+0mCdWPzHg+yWb0FnoG9c5Hofdx3ZzvvV8j6+vr1lPU1tTUk7V7K7UW8qqA6to72h35fhv7n+TaT+exr++8688OOtBdjy0g9uuua1f+xARPjPzM1Q8VMFHr/so3yj7Bjf87IZ+L75nroyFvnGd43Ho0A62123v8fWyqjJSJIWFBQujXFns8RX6ONV0qs+B73A71XSK+353H7e8eAvpKemsuncVSz+0lOGZw694n7lDc/mPu/6DVz71CnXn6pj9s9k89uZjNLY2hrFy052FvnFdX8sx+Cv9XJ93PdlZ2dEsKyYFZy9Fs8Xz292/pWhpES+Uv8BX532V8r8vD+sH8Eev+ygVD1Vwb/G9fHftdyn+aTFrDqwJ2/7NxSz0jesKRhQwMmtkj339cy3nWF+zPmmnanaXNyyP68ZcF5X5+nUNdXzy15/kY//xMcYOGcs797/Dkx94kkHpg8J+rJGDRvLsHc/y5mfepLW9lYXPL+ThPz7M2eazYT9WsrPQN64TEZw8p8fQX1u9ltaO1qS9KKsnPq+PNQfWRGzwU1V5sfxFin5UxO/2/I65LuCVAAAK/0lEQVQnfE+w4e82cP246yNyvK5unnQz2x/czhdv/CI/3vhjpv5oKn/a+6eIHzeZhBT6InKriOwRkX0i8lgPrz8lIlsDX++JyKnA88Uisk5EdorINhH5VLh/AJMYHI/DtrptlwRZWWUZaSlpzM+f71JlscdX6ONc6zk2HN4Q9n0fPH2Q2//9du757T1MHj2ZrZ/bytcXfJ301PSwH6s3QzKG8NStT/H2fW8zLHNYZz2v3MPx88ejVkMi6zP0RSQVWArcBhQBS0SkqOs2qvolVS1W1WLgaeDlwEvngXtUdSpwK/ADEbHGrLmE43FoaW9h17GL5237q/zcOP5GhmQMcamy2BO8QC2cff0O7WDpu0uZ+qOprDmwhh/e+kPWfHYNU3KmhO0Y/TVnwhw2P7CZf1r4TyzbsYwpS6ewYueKmLwiOZ6EcqY/G9inqvtVtQVYDtx5me2XAMsAVPU9Vd0b+P4wcBRI3qtrTK+CyzF0vZPW6abTbDy80Vo73YwePJqZuTPDtr7+nmN7WPT8Ih750yPMnTCXHQ/t4PM3fp7UlNSw7H8gMtMyebz0cTY9sIn8Efl86qVP8fEVH+fw2cNulxa3Qgn98UB1l8c1gecuISIFQCFwySmIiMwGMoD3+1+mSXTXjr6WwemDL+rrrzm4hg7tsNDvga/Qx9qDa2lqa7rifbR1tPHkW08y8ycz2XF0B7+48xe88ek3YvKuZDNyZ7D+/vV875bv8fq+1ylaWsRzm5+zs/4rEEroX3pNNfT2J3038JKqXnTliIjkAS8Cn1W9dMENEXlARDaKyMb6+voQSjKJJjUllRm5My6atumv9JOVlsWcCXNcrCw2+Qp9NLc3s6563RW9f2vtVm589ka+9p9f40PXfohdD+/i3uJ7e1xCIVakpaTx6E2Psv3B7RR7irn/9/dzy4u3sP/kfrdLiyuhhH4NMLHL4wlAb79b3U2gtRMkIsOBPwLfUNX1Pb1JVZ9R1VmqOisnx7o/ycrxOGyt3XphIS5/pZ+bJt5EVlqWy5XFngX5C0iRlH63eJramvjH//xHZj0zi0NnDvHSX7/Ebz75GzxDPRGqNPyuHnU1/r/185MP/YR3D73L9B9P5wfrf+DaVcrxJpTQ3wBcIyKFIpJBZ7BfcjscEZkMjATWdXkuA3gF+KWq/jo8JZtEVZJXwtmWs+w/uZ/j549TXlee9Ovt9GZE1ghmjZvVr8Hct6vfxvmpw3fe+g6fnvFpKh6u4BNFn4hglZGTIil8btbnqHi4glJvKV9640vM/8V8Kuor3C4t5vUZ+qraBjwCvAHsAlao6k4ReVxE7uiy6RJguV7cZPsksBC4t8uUzuIw1m8SiOMJLLN8ZAsrq1YCWD//MnxeH+8ceoeGlobLbtfQ0sAX/vQF5v98Pudbz/P637zO8x99nlGDRkWp0siZMHwCv1/ye3718V+x9/jezlU/V32blvYWt0uLWRJrAyGzZs3SjRs3ul2GcUFzWzND/2Uoj859lDPNZ3ih/AVOfvVkVOeIx5M/v/9nPvhvH+RPf/Mnbr361l63eeD3D3Dw9EEevuFhvnPzdxiWOSzKlUZH/bl6vvD6F1i+YznTx07nuTue44bxN7hdVtSIyCZVndXXdnZFrokZmWmZTM2ZypbaLZRVlbGwYKEF/mXMmziP9JT0Hu+be6LxxIU7VWWlZbH6s6t5+vanEzbwAXKG5LDsE8v43d2/43jjceY8N4ev/OUrva7emqws9E1McfIc1lavZdexXdba6cOQjCHMmTDnknV4flPxG4qWFvFi+Yt8ff7X2fr3W5PqiuY7Jt9BxUMV3O/cz/fe/h4zfzKTVVWr3C4rZljom5jieJwLPWpbZK1vpd5SNh/ZzKmmU9Q21HLXiru469d3MW7YODY+sJEnbn4iKWc/jcgawU8/8lP89/hRVRa/sJgH//AgZ5rPuF2a6yz0TUwJDuZmZ2VT7LEx/774Cn10aAeP/vlRipYW8Yf3/sC/3PwvvHP/O/bnB5QWlrLtwW18ee6XeWbzM0z90VT++N4f3S7LVWluF2BMV8GgWuxdHBPLAMS6ORPmkJWWxXNbnmN+/nye/cizTB4z2e2yYsrg9MF8/6++zyenfpL7Xr2PDy/7MNeOvpa0lNiLvxm5M1j2iWV9bzgAsfdTm6Q2LHMY37/l+8zLn+d2KXEhMy2Tp297GlXlvpL7SBH75b03s8fPZtMDm3hq3VNsPBKbMwQLswsjfgybsmmMMQnApmwaY4y5hIW+McYkEQt9Y4xJIhb6xhiTRCz0jTEmiVjoG2NMErHQN8aYJGKhb4wxSSTmLs4SkXrgwAB2MQY4FqZywsnq6h+rq3+srv5JxLoKVLXP+83GXOgPlIhsDOWqtGizuvrH6uofq6t/krkua+8YY0wSsdA3xpgkkoih/4zbBfTC6uofq6t/rK7+Sdq6Eq6nb4wxpneJeKZvjDGmFwkT+iLycxE5KiI73K4lSEQmikiZiOwSkZ0i8g9u1wQgIlki8q6IlAfq+l9u19SViKSKyBYR+YPbtQSJSJWIbBeRrSISMzd8EJFsEXlJRHYH/p7NdbsmABGZHPizCn6dEZEvxkBdXwr8nd8hIstEJCZuICwi/xCoaWek/5wSpr0jIguBBuCXqjrN7XoARCQPyFPVzSIyDNgEfFRVK1yuS4AhqtogIunAW8A/qOp6N+sKEpH/DswChqvqh92uBzpDH5ilqjE1t1tEXgDWqOqzIpIBDFbVU27X1ZWIpAKHgBtVdSDX4Ay0jvF0/l0vUtVGEVkBvKaqz7tVU6CuacByYDbQArwOPKiqeyNxvIQ501fV1cAJt+voSlWPqOrmwPdngV3AeHerAu3UEHiYHviKiU9/EZkAfAh41u1aYp2IDAcWAs8BqGpLrAV+wM3A+24GfhdpwCARSQMGA4ddrgdgCrBeVc+rahuwCvhYpA6WMKEf60TECzjAO+5W0inQQtkKHAX+oqoxURfwA+ArQIfbhXSjwJ9FZJOIPOB2MQGTgHrgF4F22LMiMsTtonpwNxDZu32HQFUPAd8HDgJHgNOq+md3qwJgB7BQREaLyGDgdmBipA5moR8FIjIU+A3wRVU943Y9AKrarqrFwARgduBXTFeJyIeBo6q6ye1aejBPVUuA24CHA+1Et6UBJcCPVdUBzgGPuVvSxQItpzuAX8dALSOBO4FCYBwwREQ+7W5VoKq7gO8Cf6GztVMOtEXqeBb6ERbomf8G+JWqvux2Pd0F2gErgVtdLgVgHnBHoH++HPCJyL+5W1InVT0c+O9R4BU6+69uqwFquvyW9hKdHwKx5DZgs6rWuV0I8AGgUlXrVbUVeBm4yeWaAFDV51S1RFUX0tmmjkg/Hyz0IyowYPocsEtV/6/b9QSJSI6IZAe+H0TnP4bd7lYFqvo1VZ2gql46WwJ+VXX9TExEhgQG4gm0T/6Kzl/JXaWqtUC1iEwOPHUz4OokgR4sIQZaOwEHgTkiMjjwb/NmOsfZXCciYwP/zQc+TgT/zNIiteNoE5FlwGJgjIjUAP9TVZ9ztyrmAZ8Btgf65wBfV9XXXKwJIA94ITCrIgVYoaoxMz0yBuUCr3TmBGnAv6vq6+6WdMHngV8F2ij7gc+6XM8Fgf70LcDn3K4FQFXfEZGXgM10tk+2EDtX5v5GREYDrcDDqnoyUgdKmCmbxhhj+mbtHWOMSSIW+sYYk0Qs9I0xJolY6BtjTBKx0DfGmCRioW+MMUnEQt8YY5KIhb4xxiSR/w83BOSPLBrX0gAAAABJRU5ErkJggg==\n", 
                        "text/plain": "<matplotlib.figure.Figure at 0x7f40621bb9e8>"
                    }, 
                    "metadata": {}
                }
            ], 
            "source": "plt.plot(range(1,k),mean_dec,'g')"
        }, 
        {
            "execution_count": 68, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "The best accuracy was at 0.828571428571 and max_depth of 3+1 =4\n\n             precision    recall  f1-score   support\n\n COLLECTION       0.30      0.50      0.37        12\n    PAIDOFF       0.88      0.76      0.81        58\n\navg / total       0.78      0.71      0.74        70\n\n"
                }
            ], 
            "source": "dec_acc = accuracy_score(y_test, yhat_dec)\nprint( \"The best accuracy was at\", mean_dec.max(), \"and max_depth of 3+1 =4\")\nprint()\nprint(classification_report(y_test, yhat_dec))"
        }, 
        {
            "source": "# Support Vector Machine", 
            "cell_type": "markdown", 
            "metadata": {}
        }, 
        {
            "execution_count": 84, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "execution_count": 84, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,\n  decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly',\n  max_iter=-1, probability=False, random_state=None, shrinking=True,\n  tol=0.001, verbose=False)"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "from sklearn import svm\nclf_rbf = svm.SVC(kernel='rbf')\nclf_pol = svm.SVC(kernel='poly')\nclf_lin = svm.SVC(kernel='linear')\nclf_rbf.fit(X_train, y_train)\nclf_lin.fit(X_train, y_train)\nclf_pol.fit(X_train, y_train)"
        }, 
        {
            "execution_count": 85, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "execution_count": 85, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "0.7142857142857143"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "#Train Model and Predict \nyhat_rbf = clf_rbf.predict(X_test)\nsvm_acc = accuracy_score(y_test, yhat_rbf)\nsvm_acc"
        }, 
        {
            "execution_count": 86, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "execution_count": 86, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "0.82857142857142863"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "#Train Model and Predict \nyhat_lin = clf_lin.predict(X_test)\nsvm_acc = accuracy_score(y_test, yhat_lin)\nsvm_acc"
        }, 
        {
            "execution_count": 87, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "execution_count": 87, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "0.82857142857142863"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "#Train Model and Predict \nyhat_pol = clf_lin.predict(X_test)\nsvm_acc = accuracy_score(y_test, yhat_pol)\nsvm_acc"
        }, 
        {
            "execution_count": 88, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "[[ 2 10]\n [10 48]]\n"
                }
            ], 
            "source": "print(confusion_matrix(y_test, yhat_rbf))"
        }, 
        {
            "execution_count": 89, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "0.714285714286\n0.714285714286\n"
                }
            ], 
            "source": "from sklearn.metrics import jaccard_similarity_score\nprint(jaccard_similarity_score(y_test, yhat_rbf))\n\nfrom sklearn.metrics import f1_score\nprint(f1_score(y_test, yhat_rbf, average='weighted'))\n"
        }, 
        {
            "execution_count": 90, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "0.828571428571\n0.750892857143\n"
                }, 
                {
                    "output_type": "stream", 
                    "name": "stderr", 
                    "text": "/opt/conda/envs/DSX-Python35/lib/python3.5/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.\n  'precision', 'predicted', average, warn_for)\n"
                }
            ], 
            "source": "from sklearn.metrics import jaccard_similarity_score\nprint(jaccard_similarity_score(y_test, yhat_lin))\n\nfrom sklearn.metrics import f1_score\nprint(f1_score(y_test, yhat_lin, average='weighted'))\n"
        }, 
        {
            "source": "# Logistic Regression", 
            "cell_type": "markdown", 
            "metadata": {}
        }, 
        {
            "execution_count": 99, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [], 
            "source": "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import confusion_matrix\n\nLR = LogisticRegression(C=0.005, solver='liblinear').fit(X_train,y_train)  #lbfgs\nLR\nLR.fit(X_train, y_train)\nyhat_lr = LR.predict(X_test)\nyhat_lrprob = LR.predict_proba(X_test)"
        }, 
        {
            "execution_count": 100, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [], 
            "source": "#Train Model and Predict \nLR.fit(X_train, y_train)\nyhat_lr = LR.predict(X_test)\n\n#Predict probabilities\nyhat_lrprob = LR.predict_proba(X_test)"
        }, 
        {
            "execution_count": 101, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "[[ 4  8]\n [ 7 51]]\n"
                }
            ], 
            "source": "print(confusion_matrix(y_test, yhat_lr))\n"
        }, 
        {
            "execution_count": 102, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "0.785714285714\n0.590535048737\n             precision    recall  f1-score   support\n\n COLLECTION       0.36      0.33      0.35        12\n    PAIDOFF       0.86      0.88      0.87        58\n\navg / total       0.78      0.79      0.78        70\n\n"
                }
            ], 
            "source": "from sklearn.metrics import jaccard_similarity_score\nprint(jaccard_similarity_score(y_test, yhat_lr))\n\nfrom sklearn.metrics import log_loss\nprint(log_loss(y_test, yhat_lrprob))\n\nprint(classification_report(y_test, yhat_lr))"
        }, 
        {
            "source": "# Model Evaluation using Test set", 
            "cell_type": "markdown", 
            "metadata": {}
        }, 
        {
            "execution_count": 103, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [], 
            "source": "from sklearn.metrics import jaccard_similarity_score\nfrom sklearn.metrics import f1_score\nfrom sklearn.metrics import log_loss"
        }, 
        {
            "source": "First, download and load the test set:", 
            "cell_type": "markdown", 
            "metadata": {}
        }, 
        {
            "execution_count": 70, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "--2019-02-01 03:09:27--  https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv\nResolving s3-api.us-geo.objectstorage.softlayer.net (s3-api.us-geo.objectstorage.softlayer.net)... 67.228.254.193\nConnecting to s3-api.us-geo.objectstorage.softlayer.net (s3-api.us-geo.objectstorage.softlayer.net)|67.228.254.193|:443... connected.\nHTTP request sent, awaiting response... 200 OK\nLength: 3642 (3.6K) [text/csv]\nSaving to: \u2018loan_test.csv\u2019\n\n100%[======================================>] 3,642       --.-K/s   in 0s      \n\n2019-02-01 03:09:27 (684 MB/s) - \u2018loan_test.csv\u2019 saved [3642/3642]\n\n"
                }
            ], 
            "source": "!wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv"
        }, 
        {
            "source": "### Load Test set for evaluation ", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }, 
        {
            "execution_count": 104, 
            "cell_type": "code", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }, 
            "outputs": [
                {
                    "execution_count": 104, 
                    "metadata": {}, 
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Unnamed: 0</th>\n      <th>Unnamed: 0.1</th>\n      <th>loan_status</th>\n      <th>Principal</th>\n      <th>terms</th>\n      <th>effective_date</th>\n      <th>due_date</th>\n      <th>age</th>\n      <th>education</th>\n      <th>Gender</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>1</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>9/8/2016</td>\n      <td>10/7/2016</td>\n      <td>50</td>\n      <td>Bechalor</td>\n      <td>female</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>5</td>\n      <td>5</td>\n      <td>PAIDOFF</td>\n      <td>300</td>\n      <td>7</td>\n      <td>9/9/2016</td>\n      <td>9/15/2016</td>\n      <td>35</td>\n      <td>Master or Above</td>\n      <td>male</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>21</td>\n      <td>21</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>9/10/2016</td>\n      <td>10/9/2016</td>\n      <td>43</td>\n      <td>High School or Below</td>\n      <td>female</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>24</td>\n      <td>24</td>\n      <td>PAIDOFF</td>\n      <td>1000</td>\n      <td>30</td>\n      <td>9/10/2016</td>\n      <td>10/9/2016</td>\n      <td>26</td>\n      <td>college</td>\n      <td>male</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>35</td>\n      <td>35</td>\n      <td>PAIDOFF</td>\n      <td>800</td>\n      <td>15</td>\n      <td>9/11/2016</td>\n      <td>9/25/2016</td>\n      <td>29</td>\n      <td>Bechalor</td>\n      <td>male</td>\n    </tr>\n  </tbody>\n</table>\n</div>", 
                        "text/plain": "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n0           1             1     PAIDOFF       1000     30       9/8/2016   \n1           5             5     PAIDOFF        300      7       9/9/2016   \n2          21            21     PAIDOFF       1000     30      9/10/2016   \n3          24            24     PAIDOFF       1000     30      9/10/2016   \n4          35            35     PAIDOFF        800     15      9/11/2016   \n\n    due_date  age             education  Gender  \n0  10/7/2016   50              Bechalor  female  \n1  9/15/2016   35       Master or Above    male  \n2  10/9/2016   43  High School or Below  female  \n3  10/9/2016   26               college    male  \n4  9/25/2016   29              Bechalor    male  "
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "test_df = pd.read_csv('loan_test.csv')\ntest_df.head()"
        }, 
        {
            "execution_count": 105, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "execution_count": 105, 
                    "metadata": {}, 
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Principal</th>\n      <th>terms</th>\n      <th>age</th>\n      <th>Gender</th>\n      <th>weekend</th>\n      <th>Bechalor</th>\n      <th>High School or Below</th>\n      <th>college</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>50</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>300</td>\n      <td>7</td>\n      <td>35</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>43</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>1000</td>\n      <td>30</td>\n      <td>26</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>800</td>\n      <td>15</td>\n      <td>29</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n  </tbody>\n</table>\n</div>", 
                        "text/plain": "   Principal  terms  age  Gender  weekend  Bechalor  High School or Below  \\\n0       1000     30   50       1        0         1                     0   \n1        300      7   35       0        1         0                     0   \n2       1000     30   43       1        1         0                     1   \n3       1000     30   26       0        1         0                     0   \n4        800     15   29       0        1         1                     0   \n\n   college  \n0        0  \n1        0  \n2        0  \n3        1  \n4        0  "
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "test_df['due_date'] = pd.to_datetime(test_df['due_date'])\ntest_df['effective_date'] = pd.to_datetime(test_df['effective_date'])\n\ntest_df['dayofweek'] = test_df['effective_date'].dt.dayofweek\ntest_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)\ntest_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)\n\nFeature_test = test_df[['Principal','terms','age','Gender','weekend']]\nFeature_test = pd.concat([Feature_test,pd.get_dummies(test_df['education'])], axis=1)\nFeature_test.drop(['Master or Above'], axis = 1,inplace=True)\nFeature_test.head()"
        }, 
        {
            "execution_count": 106, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "execution_count": 106, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "array([[ 0.49362588,  0.92844966,  3.05981865,  1.97714211, -1.30384048,\n         2.39791576, -0.79772404, -0.86135677],\n       [-3.56269116, -1.70427745,  0.53336288, -0.50578054,  0.76696499,\n        -0.41702883, -0.79772404, -0.86135677],\n       [ 0.49362588,  0.92844966,  1.88080596,  1.97714211,  0.76696499,\n        -0.41702883,  1.25356634, -0.86135677],\n       [ 0.49362588,  0.92844966, -0.98251057, -0.50578054,  0.76696499,\n        -0.41702883, -0.79772404,  1.16095912],\n       [-0.66532184, -0.78854628, -0.47721942, -0.50578054,  0.76696499,\n         2.39791576, -0.79772404, -0.86135677]])"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "X = Feature_test\n\n\nX_eval = preprocessing.StandardScaler().fit(X).transform(X)\nX_eval[0:5]"
        }, 
        {
            "execution_count": 107, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "execution_count": 107, 
                    "metadata": {}, 
                    "data": {
                        "text/plain": "array(['PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF'], dtype=object)"
                    }, 
                    "output_type": "execute_result"
                }
            ], 
            "source": "y_eval = test_df['loan_status'].values\ny[0:5]"
        }, 
        {
            "execution_count": 111, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [], 
            "source": "neig_eval = KNeighborsClassifier(n_neighbors =7)\nneig_eval.fit(X_train, y_train)\nyhat_knn_eval = neig_eval.predict(X_eval)\n    \ndectree_eval = DecisionTreeClassifier(criterion=\"entropy\", max_depth = 4)\ndectree_eval.fit(X_train, y_train)\nyhat_dec_eval = dectree_eval.predict(X_eval)\n\nclf_lin_eval = svm.SVC(kernel='linear')\nclf_lin_eval.fit(X_train, y_train)\nyhat_svm_eval = clf_lin_eval.predict(X_eval)\n\nLR_eval = LogisticRegression(C=0.005, solver='liblinear').fit(X_train,y_train)  #lbfgs\nLR_eval.fit(X_train, y_train)\nyhat_lr_eval = LR_eval.predict(X_eval)\n#Predict probabilities\nyhat_lr_prob = LR_eval.predict_proba(X_eval)"
        }, 
        {
            "execution_count": 112, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "KNN Jaccard Score = 0.722222222222\n\nKNN F1 Score is  0.700198920148\n"
                }
            ], 
            "source": "# KNN Model accuracy\n\nknn_jac_eval = jaccard_similarity_score(y_eval, yhat_knn_eval)\nprint(\"KNN Jaccard Score =\", knn_jac_eval)\nprint()\nprint(\"KNN F1 Score is \",f1_score(y_eval, yhat_knn_eval, average='weighted'))\n"
        }, 
        {
            "execution_count": 113, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "Decision Trees's Jaccard Score is  0.722222222222\n\nDecision Tree's F1 Score is  0.730251827026\n\n"
                }
            ], 
            "source": "# Decision Tree Model accuracy\n#print( \"The best accuracy was with\", dec_acc.max(), \"with max_depth=\", dectree.max_depth)\n\ndec_jac_eval = jaccard_similarity_score(y_eval, yhat_dec_eval)\nprint(\"Decision Trees's Jaccard Score is \",dec_jac_eval)\nprint()\nprint(\"Decision Tree's F1 Score is \",f1_score(y_eval, yhat_dec_eval, average='weighted'))\nprint()\n"
        }, 
        {
            "execution_count": 114, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "SVM's Jaccard Score is  0.740740740741\n\nLogistic's F1 Score is  0.630417651694\n"
                }, 
                {
                    "output_type": "stream", 
                    "name": "stderr", 
                    "text": "/opt/conda/envs/DSX-Python35/lib/python3.5/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.\n  'precision', 'predicted', average, warn_for)\n"
                }
            ], 
            "source": "# SVM Model accuracy\nprint(\"SVM's Jaccard Score is \", jaccard_similarity_score(y_eval, yhat_svm_eval))\nprint()\nprint(\"Logistic's F1 Score is \",f1_score(y_eval, yhat_svm_eval, average='weighted'))\n"
        }, 
        {
            "execution_count": 115, 
            "cell_type": "code", 
            "metadata": {}, 
            "outputs": [
                {
                    "output_type": "stream", 
                    "name": "stdout", 
                    "text": "Logistic's Jaccard Score is  0.759259259259\n\nLogistic's Log Loss is  0.614168569408\n\nLogistic's F1 Score is  0.671764237356\n\n"
                }
            ], 
            "source": "print(\"Logistic's Jaccard Score is \", jaccard_similarity_score(y_eval, yhat_lr_eval))\nprint()\nprint(\"Logistic's Log Loss is \", log_loss(y_eval, yhat_lr_prob))\nprint()\nprint(\"Logistic's F1 Score is \",f1_score(y_eval, yhat_lr_eval, average='weighted'))\nprint()"
        }, 
        {
            "source": "# Report\nYou should be able to report the accuracy of the built model using different evaluation metrics:", 
            "cell_type": "markdown", 
            "metadata": {}
        }, 
        {
            "source": "| Algorithm          | Jaccard | F1-score | LogLoss |\n|--------------------|---------|----------|---------|\n| KNN                | 0.72      | 0.70      | NA      |\n| Decision Tree      | 0.72      | 0.73       | NA      |\n| SVM                | 0.74      | 0.63        | NA      |\n| LogisticRegression | 0.76    | 0.71     |   0.68     |", 
            "cell_type": "markdown", 
            "metadata": {}
        }, 
        {
            "source": "<h2>Want to learn more?</h2>\n\nIBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems \u2013 by your enterprise as a whole. A free trial is available through this course, available here: <a href=\"http://cocl.us/ML0101EN-SPSSModeler\">SPSS Modeler</a>\n\nAlso, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href=\"https://cocl.us/ML0101EN_DSX\">Watson Studio</a>\n\n<h3>Thanks for completing this lesson!</h3>\n\n<h4>Author:  <a href=\"https://ca.linkedin.com/in/saeedaghabozorgi\">Saeed Aghabozorgi</a></h4>\n<p><a href=\"https://ca.linkedin.com/in/saeedaghabozorgi\">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients\u2019 ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>\n\n<hr>\n\n<p>Copyright &copy; 2018 <a href=\"https://cocl.us/DX0108EN_CC\">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href=\"https://bigdatauniversity.com/mit-license/\">MIT License</a>.</p>", 
            "cell_type": "markdown", 
            "metadata": {
                "button": false, 
                "new_sheet": false, 
                "run_control": {
                    "read_only": false
                }
            }
        }
    ], 
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3.5", 
            "name": "python3", 
            "language": "python"
        }, 
        "language_info": {
            "mimetype": "text/x-python", 
            "nbconvert_exporter": "python", 
            "version": "3.5.5", 
            "name": "python", 
            "file_extension": ".py", 
            "pygments_lexer": "ipython3", 
            "codemirror_mode": {
                "version": 3, 
                "name": "ipython"
            }
        }
    }, 
    "nbformat": 4
}