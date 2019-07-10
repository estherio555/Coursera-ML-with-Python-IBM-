{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "<a href=\"https://www.bigdatauniversity.com\"><img src=\"https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png\" width=\"400\" align=\"center\"></a>\n",
    "\n",
    "<h1><center>Multiple Linear Regression</center></h1>\n",
    "\n",
    "<h4>About this Notebook</h4>\n",
    "In this notebook, we learn how to use scikit-learn to implement Multiple linear regression. We download a dataset that is related to fuel consumption and Carbon dioxide emission of cars. Then, we split our data into training and test sets, create a model using training set, Evaluate your model using test set, and finally use model to predict unknown value\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1>Table of contents</h1>\n",
    "\n",
    "<div class=\"alert alert-block alert-info\" style=\"margin-top: 20px\">\n",
    "    <ol>\n",
    "        <li><a href=\"#understanding-data\">Understanding the Data</a></li>\n",
    "        <li><a href=\"#reading_data\">Reading the Data in</a></li>\n",
    "        <li><a href=\"#multiple_regression_model\">Multiple Regression Model</a></li>\n",
    "        <li><a href=\"#prediction\">Prediction</a></li>\n",
    "        <li><a href=\"#practice\">Practice</a></li>\n",
    "    </ol>\n",
    "</div>\n",
    "<br>\n",
    "<hr>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Importing Needed packages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import pylab as pl\n",
    "import numpy as np\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Downloading Data\n",
    "To download the data, we will use !wget to download it from IBM Object Storage."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "button": false,
    "collapsed": true,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [],
   "source": [
    "!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "__Did you know?__ When it comes to Machine Learning, you will likely be working with large datasets. As a business, where can you host your data? IBM is offering a unique opportunity for businesses, with 10 Tb of IBM Cloud Object Storage: [Sign up now for free](http://cocl.us/ML0101EN-IBM-Offer-CC)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "\n",
    "<h2 id=\"understanding_data\">Understanding the Data</h2>\n",
    "\n",
    "### `FuelConsumption.csv`:\n",
    "We have downloaded a fuel consumption dataset, **`FuelConsumption.csv`**, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. [Dataset source](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)\n",
    "\n",
    "- **MODELYEAR** e.g. 2014\n",
    "- **MAKE** e.g. Acura\n",
    "- **MODEL** e.g. ILX\n",
    "- **VEHICLE CLASS** e.g. SUV\n",
    "- **ENGINE SIZE** e.g. 4.7\n",
    "- **CYLINDERS** e.g 6\n",
    "- **TRANSMISSION** e.g. A6\n",
    "- **FUELTYPE** e.g. z\n",
    "- **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9\n",
    "- **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9\n",
    "- **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2\n",
    "- **CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "<h2 id=\"reading_data\">Reading the data in</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MODELYEAR</th>\n",
       "      <th>MAKE</th>\n",
       "      <th>MODEL</th>\n",
       "      <th>VEHICLECLASS</th>\n",
       "      <th>ENGINESIZE</th>\n",
       "      <th>CYLINDERS</th>\n",
       "      <th>TRANSMISSION</th>\n",
       "      <th>FUELTYPE</th>\n",
       "      <th>FUELCONSUMPTION_CITY</th>\n",
       "      <th>FUELCONSUMPTION_HWY</th>\n",
       "      <th>FUELCONSUMPTION_COMB</th>\n",
       "      <th>FUELCONSUMPTION_COMB_MPG</th>\n",
       "      <th>CO2EMISSIONS</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2014</td>\n",
       "      <td>ACURA</td>\n",
       "      <td>ILX</td>\n",
       "      <td>COMPACT</td>\n",
       "      <td>2.0</td>\n",
       "      <td>4</td>\n",
       "      <td>AS5</td>\n",
       "      <td>Z</td>\n",
       "      <td>9.9</td>\n",
       "      <td>6.7</td>\n",
       "      <td>8.5</td>\n",
       "      <td>33</td>\n",
       "      <td>196</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2014</td>\n",
       "      <td>ACURA</td>\n",
       "      <td>ILX</td>\n",
       "      <td>COMPACT</td>\n",
       "      <td>2.4</td>\n",
       "      <td>4</td>\n",
       "      <td>M6</td>\n",
       "      <td>Z</td>\n",
       "      <td>11.2</td>\n",
       "      <td>7.7</td>\n",
       "      <td>9.6</td>\n",
       "      <td>29</td>\n",
       "      <td>221</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2014</td>\n",
       "      <td>ACURA</td>\n",
       "      <td>ILX HYBRID</td>\n",
       "      <td>COMPACT</td>\n",
       "      <td>1.5</td>\n",
       "      <td>4</td>\n",
       "      <td>AV7</td>\n",
       "      <td>Z</td>\n",
       "      <td>6.0</td>\n",
       "      <td>5.8</td>\n",
       "      <td>5.9</td>\n",
       "      <td>48</td>\n",
       "      <td>136</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2014</td>\n",
       "      <td>ACURA</td>\n",
       "      <td>MDX 4WD</td>\n",
       "      <td>SUV - SMALL</td>\n",
       "      <td>3.5</td>\n",
       "      <td>6</td>\n",
       "      <td>AS6</td>\n",
       "      <td>Z</td>\n",
       "      <td>12.7</td>\n",
       "      <td>9.1</td>\n",
       "      <td>11.1</td>\n",
       "      <td>25</td>\n",
       "      <td>255</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2014</td>\n",
       "      <td>ACURA</td>\n",
       "      <td>RDX AWD</td>\n",
       "      <td>SUV - SMALL</td>\n",
       "      <td>3.5</td>\n",
       "      <td>6</td>\n",
       "      <td>AS6</td>\n",
       "      <td>Z</td>\n",
       "      <td>12.1</td>\n",
       "      <td>8.7</td>\n",
       "      <td>10.6</td>\n",
       "      <td>27</td>\n",
       "      <td>244</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   MODELYEAR   MAKE       MODEL VEHICLECLASS  ENGINESIZE  CYLINDERS  \\\n",
       "0       2014  ACURA         ILX      COMPACT         2.0          4   \n",
       "1       2014  ACURA         ILX      COMPACT         2.4          4   \n",
       "2       2014  ACURA  ILX HYBRID      COMPACT         1.5          4   \n",
       "3       2014  ACURA     MDX 4WD  SUV - SMALL         3.5          6   \n",
       "4       2014  ACURA     RDX AWD  SUV - SMALL         3.5          6   \n",
       "\n",
       "  TRANSMISSION FUELTYPE  FUELCONSUMPTION_CITY  FUELCONSUMPTION_HWY  \\\n",
       "0          AS5        Z                   9.9                  6.7   \n",
       "1           M6        Z                  11.2                  7.7   \n",
       "2          AV7        Z                   6.0                  5.8   \n",
       "3          AS6        Z                  12.7                  9.1   \n",
       "4          AS6        Z                  12.1                  8.7   \n",
       "\n",
       "   FUELCONSUMPTION_COMB  FUELCONSUMPTION_COMB_MPG  CO2EMISSIONS  \n",
       "0                   8.5                        33           196  \n",
       "1                   9.6                        29           221  \n",
       "2                   5.9                        48           136  \n",
       "3                  11.1                        25           255  \n",
       "4                  10.6                        27           244  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"FuelConsumptionCo2.csv\")\n",
    "\n",
    "# take a look at the dataset\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lets select some features that we want to use for regression."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ENGINESIZE</th>\n",
       "      <th>CYLINDERS</th>\n",
       "      <th>FUELCONSUMPTION_CITY</th>\n",
       "      <th>FUELCONSUMPTION_HWY</th>\n",
       "      <th>FUELCONSUMPTION_COMB</th>\n",
       "      <th>CO2EMISSIONS</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.0</td>\n",
       "      <td>4</td>\n",
       "      <td>9.9</td>\n",
       "      <td>6.7</td>\n",
       "      <td>8.5</td>\n",
       "      <td>196</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.4</td>\n",
       "      <td>4</td>\n",
       "      <td>11.2</td>\n",
       "      <td>7.7</td>\n",
       "      <td>9.6</td>\n",
       "      <td>221</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.5</td>\n",
       "      <td>4</td>\n",
       "      <td>6.0</td>\n",
       "      <td>5.8</td>\n",
       "      <td>5.9</td>\n",
       "      <td>136</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3.5</td>\n",
       "      <td>6</td>\n",
       "      <td>12.7</td>\n",
       "      <td>9.1</td>\n",
       "      <td>11.1</td>\n",
       "      <td>255</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>6</td>\n",
       "      <td>12.1</td>\n",
       "      <td>8.7</td>\n",
       "      <td>10.6</td>\n",
       "      <td>244</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>3.5</td>\n",
       "      <td>6</td>\n",
       "      <td>11.9</td>\n",
       "      <td>7.7</td>\n",
       "      <td>10.0</td>\n",
       "      <td>230</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>3.5</td>\n",
       "      <td>6</td>\n",
       "      <td>11.8</td>\n",
       "      <td>8.1</td>\n",
       "      <td>10.1</td>\n",
       "      <td>232</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>3.7</td>\n",
       "      <td>6</td>\n",
       "      <td>12.8</td>\n",
       "      <td>9.0</td>\n",
       "      <td>11.1</td>\n",
       "      <td>255</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>3.7</td>\n",
       "      <td>6</td>\n",
       "      <td>13.4</td>\n",
       "      <td>9.5</td>\n",
       "      <td>11.6</td>\n",
       "      <td>267</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ENGINESIZE  CYLINDERS  FUELCONSUMPTION_CITY  FUELCONSUMPTION_HWY  \\\n",
       "0         2.0          4                   9.9                  6.7   \n",
       "1         2.4          4                  11.2                  7.7   \n",
       "2         1.5          4                   6.0                  5.8   \n",
       "3         3.5          6                  12.7                  9.1   \n",
       "4         3.5          6                  12.1                  8.7   \n",
       "5         3.5          6                  11.9                  7.7   \n",
       "6         3.5          6                  11.8                  8.1   \n",
       "7         3.7          6                  12.8                  9.0   \n",
       "8         3.7          6                  13.4                  9.5   \n",
       "\n",
       "   FUELCONSUMPTION_COMB  CO2EMISSIONS  \n",
       "0                   8.5           196  \n",
       "1                   9.6           221  \n",
       "2                   5.9           136  \n",
       "3                  11.1           255  \n",
       "4                  10.6           244  \n",
       "5                  10.0           230  \n",
       "6                  10.1           232  \n",
       "7                  11.1           255  \n",
       "8                  11.6           267  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]\n",
    "cdf.head(9)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lets plot Emission values with respect to Engine size:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    },
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1c80016eac8>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')\n",
    "plt.xlabel(\"Engine size\")\n",
    "plt.ylabel(\"Emission\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Creating train and test dataset\n",
    "Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set. \n",
    "This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the data. It is more realistic for real world problems.\n",
    "\n",
    "This means that we know the outcome of each data point in this dataset, making it great to test with! And since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it’s truly an out-of-sample testing.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [],
   "source": [
    "msk = np.random.rand(len(df)) < 0.8\n",
    "train = cdf[msk]\n",
    "test = cdf[~msk]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Train data distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1c80221b518>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')\n",
    "plt.xlabel(\"Engine size\")\n",
    "plt.ylabel(\"Emission\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "<h2 id=\"multiple_regression_model\">Multiple Regression Model</h2>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In reality, there are multiple variables that predict the Co2emission. When more than one independent variable is present, the process is called multiple linear regression. For example, predicting co2emission using FUELCONSUMPTION_COMB, EngineSize and Cylinders of cars. The good thing here is that Multiple linear regression is the extension of simple linear regression model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Coefficients:  [[11.05909957  7.48057192  9.76190658]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn import linear_model\n",
    "regr = linear_model.LinearRegression()\n",
    "x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])\n",
    "y = np.asanyarray(train[['CO2EMISSIONS']])\n",
    "regr.fit (x, y)\n",
    "# The coefficients\n",
    "print ('Coefficients: ', regr.coef_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As mentioned before, __Coefficient__ and __Intercept__ , are the parameters of the fit line. \n",
    "Given that it is a multiple linear regression, with 3 parameters, and knowing that the parameters are the intercept and coefficients of hyperplane, sklearn can estimate them from our data. Scikit-learn uses plain Ordinary Least Squares method to solve this problem.\n",
    "\n",
    "#### Ordinary Least Squares (OLS)\n",
    "OLS is a method for estimating the unknown parameters in a linear regression model. OLS chooses the parameters of a linear function of a set of explanatory variables by minimizing the sum of the squares of the differences between the target dependent variable and those predicted by the linear function. In other words, it tries to minimizes the sum of squared errors (SSE) or mean squared error (MSE) between the target variable (y) and our predicted output ($\\hat{y}$) over all samples in the dataset.\n",
    "\n",
    "OLS can find the best parameters using of the following methods:\n",
    "    - Solving the model parameters analytically using closed-form equations\n",
    "    - Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2 id=\"prediction\">Prediction</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Residual sum of squares: 595.64\n",
      "Variance score: 0.84\n",
      "R2-score: 0.84\n"
     ]
    }
   ],
   "source": [
    "y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])\n",
    "x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])\n",
    "y = np.asanyarray(test[['CO2EMISSIONS']])\n",
    "print(\"Residual sum of squares: %.2f\"\n",
    "      % np.mean((y_hat - y) ** 2))\n",
    "\n",
    "# Explained variance score: 1 is perfect prediction\n",
    "print('Variance score: %.2f' % regr.score(x, y))\n",
    "\n",
    "#OR using sklearn\n",
    "from sklearn.metrics import r2_score\n",
    "print(\"R2-score: %.2f\" % r2_score(y, y_hat) )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "__explained variance regression score:__  \n",
    "If $\\hat{y}$ is the estimated target output, y the corresponding (correct) target output, and Var is Variance, the square of the standard deviation, then the explained variance is estimated as follow:\n",
    "\n",
    "$\\texttt{explainedVariance}(y, \\hat{y}) = 1 - \\frac{Var\\{ y - \\hat{y}\\}}{Var\\{y\\}}$  \n",
    "The best possible score is 1.0, lower values are worse."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2 id=\"practice\">Practice</h2>\n",
    "Try to use a multiple linear regression with the same dataset but this time use __FUEL CONSUMPTION in CITY__ and \n",
    "__FUEL CONSUMPTION in HWY__ instead of FUELCONSUMPTION_COMB. Does it result in better accuracy?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Coefficients:  [[11.30058699  6.75282144  7.04296941  2.17179404]]\n"
     ]
    }
   ],
   "source": [
    "# write your code here\n",
    "from sklearn import linear_model\n",
    "regr = linear_model.LinearRegression()\n",
    "x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])\n",
    "y = np.asanyarray(train[['CO2EMISSIONS']])\n",
    "regr.fit (x, y)\n",
    "# The coefficients\n",
    "print ('Coefficients: ', regr.coef_)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Residual sum of squares: 606.76\n",
      "Variance score: 0.83\n",
      "R2-score: 0.83\n"
     ]
    }
   ],
   "source": [
    "y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])\n",
    "x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])\n",
    "y = np.asanyarray(test[['CO2EMISSIONS']])\n",
    "print(\"Residual sum of squares: %.2f\"\n",
    "      % np.mean((y_hat - y) ** 2))\n",
    "\n",
    "# Explained variance score: 1 is perfect prediction\n",
    "print('Variance score: %.2f' % regr.score(x, y))\n",
    "\n",
    "#OR using sklearn\n",
    "from sklearn.metrics import r2_score\n",
    "print(\"R2-score: %.2f\" % r2_score(y, y_hat) )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Double-click __here__ for the solution.\n",
    "\n",
    "<!-- Your answer is below:\n",
    "\n",
    "regr = linear_model.LinearRegression()\n",
    "x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])\n",
    "y = np.asanyarray(train[['CO2EMISSIONS']])\n",
    "regr.fit (x, y)\n",
    "print ('Coefficients: ', regr.coef_)\n",
    "y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])\n",
    "x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])\n",
    "y = np.asanyarray(test[['CO2EMISSIONS']])\n",
    "print(\"Residual sum of squares: %.2f\"% np.mean((y_ - y) ** 2))\n",
    "print('Variance score: %.2f' % regr.score(x, y))\n",
    "\n",
    "\n",
    "-->"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "<h2>Want to learn more?</h2>\n",
    "\n",
    "IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href=\"http://cocl.us/ML0101EN-SPSSModeler\">SPSS Modeler</a>\n",
    "\n",
    "Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href=\"https://cocl.us/ML0101EN_DSX\">Watson Studio</a>\n",
    "\n",
    "<h3>Thanks for completing this lesson!</h3>\n",
    "\n",
    "<h4>Author:  <a href=\"https://ca.linkedin.com/in/saeedaghabozorgi\">Saeed Aghabozorgi</a></h4>\n",
    "<p><a href=\"https://ca.linkedin.com/in/saeedaghabozorgi\">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>\n",
    "\n",
    "<hr>\n",
    "\n",
    "<p>Copyright &copy; 2018 <a href=\"https://cocl.us/DX0108EN_CC\">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href=\"https://bigdatauniversity.com/mit-license/\">MIT License</a>.</p>"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  },
  "widgets": {
   "state": {},
   "version": "1.1.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
