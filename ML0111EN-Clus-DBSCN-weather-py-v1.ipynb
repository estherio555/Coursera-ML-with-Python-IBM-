{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a href=\"https://www.bigdatauniversity.com\"><img src=\"https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png\" width=\"400\" align=\"center\"></a>\n",
    "\n",
    "<h1><center>Density-Based Clustering</center></h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Most of the traditional clustering techniques, such as k-means, hierarchical and fuzzy clustering, can be used to group data without supervision. \n",
    "\n",
    "However, when applied to tasks with arbitrary shape clusters, or clusters within cluster, the traditional techniques might be unable to achieve good results. That is, elements in the same cluster might not share enough similarity or the performance may be poor.\n",
    "Additionally, Density-based Clustering locates regions of high density that are separated from one another by regions of low density. Density, in this context, is defined as the number of points within a specified radius.\n",
    "\n",
    "\n",
    "\n",
    "In this section, the main focus will be manipulating the data and properties of DBSCAN and observing the resulting clustering."
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
    "        <li>Clustering with Randomly Generated Data</li>\n",
    "            <ol>\n",
    "                <li><a href=\"#data_generation\">Data generation</a></li>\n",
    "                <li><a href=\"#modeling\">Modeling</a></li>\n",
    "                <li><a href=\"#distinguishing_outliers\">Distinguishing Outliers</a></li>\n",
    "                <li><a href=\"#data_visualization\">Data Visualization</a></li>\n",
    "            </ol>\n",
    "        <li><a href=\"#weather_station_clustering\">Weather Station Clustering with DBSCAN & scikit-learn</a></li>   \n",
    "            <ol>\n",
    "                <li><a href=\"#download_data\">Loading data</a></li>\n",
    "                <li><a href=\"#load_dataset\">Overview data</a></li>\n",
    "                <li><a href=\"#cleaning\">Data cleaning</a></li>\n",
    "                <li><a href=\"#visualization\">Data selection</a></li>\n",
    "                <li><a href=\"#clustering\">Clustering</a></li>\n",
    "                <li><a href=\"#visualize_cluster\">Visualization of clusters based on location</a></li>\n",
    "                <li><a href=\"#clustering_location_mean_max_min_temperature\">Clustering of stations based on their location, mean, max, and min Temperature</a></li>\n",
    "                <li><a href=\"#visualization_location_temperature\">Visualization of clusters based on location and Temperature</a></li>\n",
    "            </ol>\n",
    "    </ol>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Import the following libraries:\n",
    "<ul>\n",
    "    <li> <b>numpy as np</b> </li>\n",
    "    <li> <b>DBSCAN</b> from <b>sklearn.cluster</b> </li>\n",
    "    <li> <b>make_blobs</b> from <b>sklearn.datasets.samples_generator</b> </li>\n",
    "    <li> <b>StandardScaler</b> from <b>sklearn.preprocessing</b> </li>\n",
    "    <li> <b>matplotlib.pyplot as plt</b> </li>\n",
    "</ul> <br>\n",
    "Remember <b> %matplotlib inline </b> to display plots"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Notice: For visualization of map, you need basemap package.\n",
    "# if you dont have basemap install on your machine, you can use the following line to install it\n",
    "# !conda install -c conda-forge  basemap==1.1.0  matplotlib==2.2.2  -y\n",
    "# Notice: you maight have to refresh your page and re-run the notebook after installation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "from sklearn.cluster import DBSCAN \n",
    "from sklearn.datasets.samples_generator import make_blobs \n",
    "from sklearn.preprocessing import StandardScaler \n",
    "import matplotlib.pyplot as plt \n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2 id=\"data_generation\">Data generation</h2>\n",
    "The function below will generate the data points and requires these inputs:\n",
    "<ul>\n",
    "    <li> <b>centroidLocation</b>: Coordinates of the centroids that will generate the random data. </li>\n",
    "    <ul> <li> Example: input: [[4,3], [2,-1], [-1,4]] </li> </ul>\n",
    "    <li> <b>numSamples</b>: The number of data points we want generated, split over the number of centroids (# of centroids defined in centroidLocation) </li>\n",
    "    <ul> <li> Example: 1500 </li> </ul>\n",
    "    <li> <b>clusterDeviation</b>: The standard deviation between the clusters. The larger the number, the further the spacing. </li>\n",
    "    <ul> <li> Example: 0.5 </li> </ul>\n",
    "</ul>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def createDataPoints(centroidLocation, numSamples, clusterDeviation):\n",
    "    # Create random data and store in feature matrix X and response vector y.\n",
    "    X, y = make_blobs(n_samples=numSamples, centers=centroidLocation, \n",
    "                                cluster_std=clusterDeviation)\n",
    "    \n",
    "    # Standardize features by removing the mean and scaling to unit variance\n",
    "    X = StandardScaler().fit_transform(X)\n",
    "    return X, y"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Use <b>createDataPoints</b> with the <b>3 inputs</b> and store the output into variables <b>X</b> and <b>y</b>."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2 id=\"modeling\">Modeling</h2>\n",
    "DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. This technique is one of the most common clustering algorithms  which works based on density of object.\n",
    "The whole idea is that if a particular point belongs to a cluster, it should be near to lots of other points in that cluster.\n",
    "\n",
    "It works based on two parameters: Epsilon and Minimum Points  \n",
    "__Epsilon__ determine a specified radius that if includes enough number of points within, we call it dense area  \n",
    "__minimumSamples__ determine the minimum number of data points we want in a neighborhood to define a cluster.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 1, ..., 2, 1, 1], dtype=int64)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "epsilon = 0.3\n",
    "minimumSamples = 7\n",
    "db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)\n",
    "labels = db.labels_\n",
    "labels"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2 id=\"distinguishing_outliers\">Distinguishing Outliers</h2>\n",
    "Lets Replace all elements with 'True' in core_samples_mask that are in the cluster, 'False' if the points are outliers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True,  True, ...,  True,  True,  True])"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# First, create an array of booleans using the labels from db.\n",
    "core_samples_mask = np.zeros_like(db.labels_, dtype=bool)\n",
    "core_samples_mask[db.core_sample_indices_] = True\n",
    "core_samples_mask"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Number of clusters in labels, ignoring noise if present.\n",
    "n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)\n",
    "n_clusters_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{-1, 0, 1, 2}"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Remove repetition in labels by turning it into a set.\n",
    "unique_labels = set(labels)\n",
    "unique_labels"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2 id=\"data_visualization\">Data visualization</h2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.61960784, 0.00392157, 0.25882353, 1.        ],\n",
       "       [0.99346405, 0.74771242, 0.43529412, 1.        ],\n",
       "       [0.74771242, 0.89803922, 0.62745098, 1.        ],\n",
       "       [0.36862745, 0.30980392, 0.63529412, 1.        ]])"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create colors for the clusters.\n",
    "colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))\n",
    "colors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYAAAAD8CAYAAAB+UHOxAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAAIABJREFUeJzsvXmUXPd13/n5va3Wrt4bOwiQBLiANEUSkijJ0Vih6CgaWfIkji2Pc2zpOEcTjz3jY53MHOdkxktm5hxvI41nrCySYstycuJFcWw5pixrsyVHpESQFC0CJEGQ2BpooNfqrvWtv/njvldVvaKBbjS6id/nnEZXV71671Wh6t7fu8v3Kq01BoPBYLj9sG71CRgMBoPh1mAcgMFgMNymGAdgMBgMtynGARgMBsNtinEABoPBcJtiHIDBYDDcphgHYDAYDLcpxgEYDAbDbYpxAAaDwXCb4tzqE1iLkZERfejQoVt9GgaDwbBjePbZZ6e11qPr2XZbO4BDhw5x4sSJW30aBoPBsGNQSp1f77YmBGQwGAy3KcYBGAwGw22KcQAGg8Fwm2IcgMFgMNymGAdgMBgMtynGARgMBsNtyrYuAzVsPTqJoDGBDhsotwSlPfLAkvuUZT46BsNOx3yLDR10ew598WsQteRvAGWB1tlf8q9TgAPvQuUHb9GZGgyGzWBTQkBKqd9WSk0qpV5c5fHvU0rNK6W+k/78wmYc17B+dBKhaxdJZl9G1y7KSn/p4xe/JsY+PyQ/uQGoXYD6uNzO7tcaffFrJFF7zX0aDIbtzWZdAXwG+C3gs2ts8w2t9fs26XiG62DFlf3SVXxjQh7PD3WfGNYBJbeDOuQqctstyvan/wCtnNX3aTAYtjWbcgWgtf46MLsZ+zIs5lor93U9f+nKvmcVn+1Ph43lT46D7u2k57ZOoHEFonDNfRoMhu3NVuYA3qaUegG4DPwzrfXJLTz2jmRdK/drsdLKHmQV356Vx/sOoNxSGuXvwfa6t62e20EddAy5vjX3aTAYtjdbVQb6HHCH1voh4P8D/mS1DZVSH1FKnVBKnZiamtqi09t+rHflfs39rLSyX+nx0h5xLmGz+6BbJkv+4pW79wc1UHb6+PUf02AwbA+2xAForRe01vX09pOAq5QaWWXbT2qtj2utj4+OrkvR9I1JtnJ3i4vvd4tyf2NiXbtRbmldjyvLQR14Fyglq/j2LPhV6DsI5f1yO7vfdsVhqJU/Ptc6psFg2B5sSQhIKbUbuKq11kqptyCOZ2Yrjr1dWanevre2fj0rd7WeA/Wu7HudSdiU+7M6f5Cw0l3vv2YfgC6MwtknJRREIrkC2wOsZfs0GAzbl01xAEqp/wh8HzCilBoHfhFwAbTW/wb4IeCnlFIR0AI+qLVeFnK+XejG9psQttA6BLeEPvRerKJc9awYk+9hvatsZTlw4F1yvHZPnt4poA68a1lDl7IcyQlk57qCo7Ish2TsETj9RxC3u0+283D0H5kmMYNhh7Ap31St9Y9e4/HfQspEb3s6sf2oDc1JyGL57Sqc/AzJsQ9jFUdWXrnrBJrToEN0EkMSrcvYrrayX/rcpcZe23m49I1lSWi97+/A5HNQOSTnlASSJFYWTD6H7ttvnIDBsAMw39KtpjEhK//mpFhUp9B9LKjB2T9H3/djy1fusS+llygo7YbL/xV9HRVBS1f2S1lWcZSVepb3LekNaMK5JwELCiukcUwVkMGwYzBicFuMDhsQtmTlb7uLH7QcCBvdBK/XB2OPiBEOm3JVMPo94gA2UHe/tLcgidrLK44sT1b2tQvQnIJgQVb7blHOMXUUq75Gg8Gw7TFXAFuMcksS818NyxED2p5DX/wqtGfAr0MwLwY5NwBOXrZdpe5+rQTziiv9oC77Lu4WI68s6QIOG+IUkkiSvJYDlcOgXEhWfw2mCshg2BkYB7DVlPaAW5KYfy9xKAbWKYCdQ5/7C1l9oyRfEAdi7JMYhu9fVIKZVQTpJELPnYbL3xRD7hTQyuo0j+H1oS98BYIGWKnIW2sK2nMSYgpq4Bag7w65XyvZznJlH3EAc69I/b+yxXF4ZTlWUJfn2y66MLq+CiWDwXBLMQ5gi1GWgz70Xjj5GTGYWbLUcqCwSwxwHMLcaVAOODkxvkkAWFKPHyyAV5FVejAPYZ2kOQWXvg6zpwGdhpNqkB+F1iz6tc/DyPfIfi1HjL9fldW9ZUtnb9SQqp6oDdhgO7LSV7b8DuYh8sURlPbIvuycPA8l+y3tgbNPoo0mkMGw7TEO4BZgFUdJjn0Yzv65OAEdi5HVkRjps1+Q8IvtQdxCBNmU/EoSaM2kyeS0BHP6JFz4moSHLDt1GKFcMbRmxVnEPsy/Lsfx+uS5SSj7zcI+WsnvYAHcPnBKEC5IuChKheGs9Pmtqa5TsAviuAbulmO3ptFn/hh94HGUqQgyGLYt5pt5i7CKIySH3ysVNWFD4upKwZn/LEZa2RJ6AQn76EQMbhxC7aIYdbcgpZixn/YU1OU5dk5W91igEjHaiSUrdx1AVID2vOwTC9DpMZVcEcQR6DoUhsQRRC1xPJYtjqNxJb1q0HJuriO/Z1+RY5PIOZ3/S3RhxCiEGgzbFOMAbpBrdfKu6/mXvi6loG6frPYzI0/UNcaQGuU2srGScFBYg9ygxN5rF8Qw6wRopwleLcY4SboOxHbBb6UlqFG6v0R+6UQMuk6AGLSVXj2kej+W1U386hhUWiVEDP5cmiNoS/VQrj/dlzgIffFrcNf7zZWAwbDNMN/IG2AzVDr13GmYeYWOlAJI+MVOG6rsXFp6mRpp2UAMr45llb7welelU1lprD6WqwQd0Vndaw3YYqiJ5fnLSOTqoHOCMegQ2j5p7Kl7DjpKZwVkvcqp49DIyr89DTgSJooa4qhMb4DBsO0wfQDXyWaodOokgot/JQber4qRjBqyqm/NpBU5zdS+9gpCpMZXIQlarYE4DfGEaUJXyX0kaVLXl8fCBeB6+gUyg586D2X3/L30vLQchyyXgCSEvYrcbkyg/YXrOLbBYNgK3tBXABsN06zINfT1de1ip5Z/VcmF2sVUBiKt7LFsxLhrIAR/FlRekq1hM12ZK9lWpWEgnQBaQjyOI8NZiABPNHmillQRuSUIqqus+tdLBNpBHMtqJD23VbdsVMfiHML6Bo5vMBhuBm9YB7Apw1RW2u9aXa6xDxe/Kho6ax1z7rQkfklX8ElCp4yys7BO0tW+Su+zZFtNGvaJ6WychPKwXZJ8go4lhJRV9GzI+GevZI3mtV7snDiyqCkVTDqRn+m/RQ8eNclgw6YR+wFzJ8/hzyyQG64weOwQds679hMNHd6QDmBZmCYjbG44IbmqSuca2jm9x9RJJA4A0tJPTScZm41dVK50+9q5dGUPi1bfOpRqH0uJM7Bc6Op3yrlYNuIVXKAntn8zUXY3ka1csO1uVZLlmWSwYdOoX7jKyY9/jmCuni6SNN5gmWM/90OUD+661ae3Y3hj5gA2aZjKiqw0OQtEpRO1XCDNyUN7Gn35KZnpWxtPZRXSmLrlyI9ySIP7cltZIgOhVzDeOq0KcsrSKObPp01hdWhchtYk+DXp8L2uuP9GSbuGvT7ID4DXD1ZenJztLdY5MhhukNgPOPnxz6FjTfnQbsp37KJ8aDc61pz8+OeIg3VerRreoFcAmzVMZQVW1dfXIRTHuro6VqqdUzsvhi+O0LUL3dh8fkhE1nRMN6afyGpG6Z4QkaZTzYOWlbWOZB+Wkv05BcCGaL57Pkk73e8NvtDrxpIf5cj5JZGEgUCkI2oXIYnQtcvp27WJeRnDbcXcyXMEc3XKh3Yvuj831Ef93BXmXjzLyCNHb9HZ7SzekN+8zRqmsurzV9DX1+05eP2/gJUOOtM67ZLNS2gkPyiduo1JWaUP3C1JUn9enEdn507agVtLjbfqub+QGtc0yWqnUtJ2Qfa5jK2cuZOA9iG2xPirtpyXV07zEUquSC5+Gd13EJS1aXkZw+2FP7Mgn6eVUEoeN6yLN2YIaLUwzQpjEG8UZTmovgNYQ/fK/qqvSlhHORIGCRZENydYkNV8/bKs1osjsuqfe6UnB5Dt1BWp50zoLSu9tPNSzWPZ3YYt0isFraVqaFEVzi0kiegMiYka6SzhaXnvs0Yyy7vhIfcGQ264svh704vW8rhhXbwhHcCKA87bs6DUimMQN0ymyzNwVBbsrel0FZxq+OQq8nvhXNeAx6HU/PfG6HUq8xDMp4Y0DfvEfmpUM5mGUGL+zem0j6DJ1q7210CHLO4hSM/bnxfna7ndZLdOK52aV9BT3zFOwLAuBo8dwhss48/WFt3vz9bwBssMPnD4Fp3ZzuMNGQKC9Y9B3Aw6OQcnL1cD2ZVH2EhX8UpkGKKWVApl2j0rrtp7qoGsXBr612kjV+9Bo7TbdzuSXqFgp9VArji8JElTBZ44zIWzqVPzYeLb6IULJhxkuCZ2zuPYz/0QJz/+OernriyrArI999o7MQCbNxT+t4H3AZNa6wdWeFwBvwm8F2gCH9JaP7cZx17zvK4xBnHTjtObc8iGp9h5MeRxLCGd2BeH0JpNE7TrIAnTsM9OrGrQqZMKIU7EAcRtcEakGqv6KotGYuaHO+EgUyr6xuJm1OuXD+7i+K9+hLkXz3b3+8BhbM9d9Ximb2A5m/Ut+wwy9P2zqzz+94Ej6c9bgX+d/n5j0JtzsDNtHiXVL8m8JD+zuPj1xur1NontZ6j0I7Ouq4+0exnSvEY6XSysy2+n0B2E45XT0lczU/iNxEbq9TOD3ZyYIZhvkOsvU9gz1DHctucuq/ZZ7XiHf/j7OPuHf2X6BpawKQ5Aa/11pdShNTb5APBZrbUGnlZKDSil9mit3xBF4YtKQ+NUkyeoSY2+lyas4jaEacx7vU5gaZJ4O7DusJMSVVAnL47RUlAYlrxH9bU0IawlJ1A5tOKEM8POZmm9foY/W+Pkxz/H8V/9yKrhmsyQN8anqJ46T+KHKNehuHcEr7/IXT/+/RR2DxLVWuSGK1Tu3svcqfO8+Gt/gJVzqdy9D8uWIorWVJVv/vRvMvb2Y9d9Hm90tuo6ex9wsefv8fS+ZQ5AKfUR4CMABw8e3JKT2wwW5RwaV2HmZGr0J1N5BBccD8L1GlAr7RHYZlcA60bLSt+fT+fZOJLg1mk4KJsn0HdHNwyUYmYKby9uNHSyWr2+219k9oXXOP3pJxl97L5l+8scRxxGNMancMtF6IP6uSv40/O4/UUmfvLXKY4NMvDAYdCahdcvkxuq0Dh/FaeUp37uCmOP3Y/XX0ZHMdFCEx0ulkQxfQNb5wBWWtCtrKig9SeBTwIcP358my1/V0YnkXT41lIf17cf7vkRuPKMyDPnh4EEZl7i2gbdSrfZqYY/I+1mVjGLhs6gUsXStEmtdh4G75FtVyjTvSmCfoZ1s5EQzkr1+sF8ncmnT9GemiduBYx/4dtYrs3Rf/Je9j7+CHbO6zgOK+cStwLcSpHa2SsoW64Sw1oLy7LAsWhcuJqK4mqal6awS3m8gTJxK2Dy6VPsffejRM02Wimilg9AEsW0p6pEzTbBfJPWxOzSU79upxf7AdPPv8rsc2cAzfAjRxl++O5tn2PYqm/SONAb1N0PrNS5tGPoGKbGFZj6Tk9XL2LoKodg7CEJg+QqsvK1nVRTLTPyisV+0Op5bIejkEY3ZUNnJkFqDDTguPJn1iPhFsEpdMp0ZcD9KzDxdCpuV0QrZRrHtpCNhHBgeb1+EsdMPn0KErBzLq0rsyjbIgkinvlf/i17Hn+EB//ZD3ccR9Ty5XejjY5i7JxL6LfQSYLtuSjLwq/WQUNx7zDNyzPELR/dXyKJY/zpGpPfPInlOcTtAMtzxAE9dYq4LZV2Ub3Fa//hKww+eLjj0K7X6dUvXOU7v/xZJp86RRJGaK2xcy6jj93Pw7/449s6x7BVDuDzwM8opX4fSf7O76T4/9JVqLbzcOkbYrxqF6XZy3JkpW85EuKoj0v5o52Tla1bhOKetPO3t04euo4gVQd9I9AZG5DKVvc6vDiWPMnAHrmr/xAqbahTliMD7s/+F1g4D9oCNwdWHSqHQSemUmiL2KjkQm+9fm6oj/ZklbgVoFwHf7ZGcf8obkmUc4NqnfZUlZMf/xyHP/gu0Brbc4mabZIwIgkjrJyLTjTKkisBy3WIGq2Oj7ELOeJmi/nTFyFKCOtN2pNVLNcGDVPfehnLtVG2hbItokYbu5jD6S90HBpar+r0vvvrf8Bd//jdhAvNzlUBwHd//Q+Yfu5VnL4CTiEHQNwKmHnuNN/9jT/kLb/xT7dtjmGzykD/I/B9wIhSahz4RUSGEq31vwGeREpAzyBloB/ejONuBctlpTXUJ6C0V+rZtRajjyUNYF5fWimjpfpn3/fC7MvyWNRKa/tjCQ0pla6Q07+BN8TqHxBHls46yPoaFk0QC6UUdOge1NC9qLTqJ2lOwcnPSJ9E5Mvzw0gqqhbOSrjIr5pKoS1go5ILS+v1GxMzYnTzHrnBvo7xz/ZnuQ7BXB2tNdgWU8+8TDBbI45iknZA7AedkIpybJxSnrDWlBEZAGiUbcsspKaP1uDkXCzXITfcR+yH1M9fwbJtkjjBsi0Ke4aYe/41SvtHmXvxLMCKTk/Zist/+Tz1c1fJj/R3rgr2PvEo9XNXQeuO8QewC1J2ut1zDJtVBfSj13hcAz+9GcfaSlaUlQ4WpIKldTWtXe8Z2K5jCHQ6FSsRY9+ahdI+mHpe7nfydIakpHMDuuMV32iN2dnQ+VXQIVTPoNX75NogieDsk/Je2vm0D8IV5xjWRQ4jHSxjKoVuPpshudBbrz/19CnGv/AMbl+B6qnzy/bnFHIkQUQwW5O6AcsiPzZA8+ocoEnCBMvV2DmH/OgAcTvArZSImz71C5MoS+EUchT3DVM/ewXb8Sgd2IXbVyCcb+ANVmicuwI5jZ3zsBybsNbEzrlUXzpP68oslussc3pZ6EonCTpJSKIIp5gnDiJOf/pJora/8otXitgPt7U2kbmGXoul0790Au1qWsveAjdMm7zCdDWf1r1brnS6NiZg+sVUqwcx/oNHYO5ViY8n6fM7uYOtVO+82Wjk43UNp6Y1nP8i+r4fk/crbKTy2D3Py0ZexlGaV7BMpdAmslrCc2kIJ+O6JRdSJ1K6YxdupUDkh0QtH601luugLIVd8MTYX5zCr9bRccK+97yZhdPj5HcNkvgh7el5oqZP/9H9tK7MEtWaxEFE7AcS4897tGcWcMtSVZYfGUBZqpOCqr0uaUenmEfZthhzP6R5aQblWMw8f4a9jz+yzOm1J6uE80386QV0rHGKOXSS7tSx02Os8MVNcwHbWZvIOIA1WCQrnUkXRM1OOIhm+p+udSdZJKt7nRp3p6v3nzU91S+nQ1I8RCIiJ6vabS3tcCMoKO8GHKifW32zrDEszbFIOC39culEDL6VxU/Tq620Gsiwca6V8Nyo5MLS/QfVBlPffpkkkEWTZVlYOVeM/auXSYIAf3qeYKFJ9eULxG2fqBWi4xg759F/dJS9TzxK5e59PPsv/h0o8Ab6sHMOcSsgqDUIa02wIJirEczVUI7dPVdLSdjIstCJJgkjSBIsz2Xiq8/TujILtrXI6YX1Fq2rcnVQ2D1IEkQ0Lk0TtwN0GOFUiiTtECyF118GIGq2iVo+ylIkYbQofAXbZ5qZcQBr0JF40IkYfw24FTFKcZxq34c9cW4tsgfybMiVpBcgw3YlCYwWwx/73eftCHrmEqyFcsAuy0Ab7JU26O5DWZDE3QS7smWmQnZVlPjyPqls2Ez55gj63Yasp8pnLcmF691/EsdUXz5Pcc8QiR9JJbDW6Djm4uefIr9rgMEHDnPpiyeYevolCrsHCRYaUr+vFEG1TuvKLPf+1A9w/o//BqUU+dEBSRC7Dsq2aIxPoS2Fk/PAUiIN0Q7wm22cciFdWCiSKEYH3QVXEsW0rs5RPjgmzgHdcXqN8Ul0nFA+Iu9R49I0JAluKU+40GTw2CEaFyZpXJgiHGyh45hgoUl+qILlurzyr/9skVPdTtPMzLdoLTKJh0zdM2tYcsuga/KfF0VpmWI6wtFy5EMWteSqoZjvDkaBbgWQV5GQh0qngW14bu9WcK0EdTanoAT+VHrXSg6gVwLbShvm6ujyfpGCsAti+HUseZRMJfTOD6BGHzDGf5NYb5XPSpIL17v/JIolzj5ZJTdQBhSjb7kXjWbymydxK0VGH7sP23UJF1okYUTt7ATeYB92Pk382hZRy+eVTz1J1Gjjzy4QVNOcUJwQNtroOJEaC0IJC+UcLFd+/NkaCiU+IFhyta01SRBSffkClSP7Ofpjj8tzZhaYO3We9tQ8aBaVpCZRjHIscsMVRo7fw8zzrzJw3x1MP3uawsgAlXv2d7qRM6f68L/80IZKazcb801ag47Ew5k/llVohuXA6ENioGrjQAS5oa4OUOzLQPSs/DEO5YrBdlJjloCfpKWjQ+kUsV4Z5Z2KkhGQmfHvTDpbY/sklLj/7MvSPe31pQPl07i/TuS+wgiqOGyM/yZyswerZPvPau/b0/OE1TpRoy11+JbCLRfQUYKTzzH1zVNorQlrsopOgoi42UbnPHQkVwG5gTKtyVn8qXkJwec9tNbiCLRGa41TylPcO4yOE0g0Q2+6m+kTr4BlkbRDokZz2bmqNOxjF3IE1TrhQpN9TxwHJBk++TffpTEu+YkkCEFrlGuTHx3ALRWwbJvC6CB9h3fjT82v6lQv/NlT22qamfk2XQOVH0QfeBzO/2WqZ+91hcsA3ALYeySOHTbTUEU63zeJoHFVtgvmelb52QrYTqUSbLrGf2lz2E7CFg0kQAbWpO+RzvoAMqxuwltZEtPPDUp5bXsWRh5MB8gE3ffbr5rKn03mZgxW6Y1tt6aqJH7I5FNi2L2BMuF8g6jRIm4FzL90HuW66DBEKYVdyGHnXKJ2QBzIPIwkSsCOyGTRW1PzWHN1LNfGzssqXKpzNMqxIYikemh0AJWGjXQiJZ/Db7qLqadfWv7tUnIFEdSa2DML5Ef6aU1VufSlEx2dodL+UfK7BmlNzDL7wmt4A2WUpVC2VCpl75l87ld3qo0LkzfV6V4vxgGsA9W3H10Ykf/g3kHzYVPi0q3JtHJFpWWhVnesI0iIQ9k9DiCt/SdOE8q9xj/7cOxAJ2Cl4yAXvQ7S9yVzbBY4RcmjOG5aSpsmgnN90JyQ9zW33PiYyp/NZb1VPr1GXeUcGmev0pyYoXRwjIPvewyvIonPpbHtJI6Zeu40xAnFfSMkcUzY8kn8ECwLy3WwCy5BGElIRkHspyKBWktKrdFGxx6gxcjbFmEYohqavrv2EtaakoyN09nalsIbLKN6jGxzYhYr5zL44GHmX7pA1A7QUbJ4raWBOKE1MYM/W6M1VcXrL8k5eh57nniY2WfPYOc9cVKNNt5AibHH7sey7c57NvzoEaa//fLKb7jWlA6OsfDyxRUfTuJ4kePZisSwcQDrYNVB8E5ejJntgHK7ziFoyACXwohIQdQup6GQ1PCrVPJBpx/2DutIsG5bsnBPz7dq0UvJ2jVzUNwN4Ty4fd08ShxIOEylVxG9DmATR3kauqxnsEqvUW9NVWUFnSR4AyUs2+bF3/hD3v6Jn2X4kSMrxrabl6eZfuYVkjhGx7qzLLDdtAwzijthwsSPwLZkgZz0SEiksg3KtsCycEsFkiCkPVmluHdY5CSiBKfg4VaK2DmP5sQMSRQTNdoU943gDfZx9a//Nl2kpAuuXuNvdY+Z+CHzJ8+Jk7ItlOcw/czL7P7eB7HLOY5+5L9l+luvoByLoNogmKt33rPCrsE1nerBH3gbM8+8suzx+nkJ/ygtTW5blRhWervJDfdw/PhxfeLEiVt9Gh2WSUIkMVz+r+IEsulWIMnfqAnD9wNKJA3CRrdZDOT+ZTH/Hbz6J9X8UVZazrpKKMutQHmPNNQ5qcOMWlC5I02MX+kJo6VkGkFG/+emEAfhqoNVTvz8p9Cxxi7mOPO7X5TafccGBX137iWsNsBWPPb//gyvfvoLi4x/MF/n8pefo3b+KrmhMokfEczXJSnrueg4Jj82SOvyNFE9rZaz1CLjvwgFKu/hlQvEQYSyLSp37sEu5miOT2PlXVnZa03UaBPMSxl3ftcg/uQcKEXpwBitK7OE9dbyRPAq2KU8ludQ2jvC7nc+hOXZPPx/fJiF0+MrVkZlTtOfXsCfbxC3fPJj/Tz8Sx+i/8j+Fa+U5l48y/BDd1O+o2vs/dkaylbXnRhWSj2rtT6+nm3NFcB1sHTCmJ59uVurnh+RFb1yxADWLiOXpF43EZrN+F3R+IP8d+zUXgARbEMB0Wpqppa8V7WLchvV7ZVwy2nopx8OvxfVmjIKoFvEalU+vVU8s999nTgI8frEacftQMIgg2Wa41OM//m3FsW2kyhm8qlTWHkPJ+9iOVKnn8QJSeCjwgjLtrEcG+U63fXCasaf9PEoxq/WpVQzjqmePIflOSjPJRxvSG+BYxEHEZZjU9g1SHtiltgPULbNwqvj4nyidVbdKYjbPpbnELV8kigiarRYOD2+arK2fHAX9/70D/L8L3xG6v8LHsq2eflf/WlnRd9bWtuaqqI0i4w/bE1i+I2mPbC16FiMWe08NK/IsJP2DHj9qd59IgnMrOmr8/3oFYHrTQiFrP5fst3Sn9m5pxLPmf6RV5Ewj8pWLGm4yypIvD+t+wdLtJLitiSA/SoohTrwLiwnj+o7gJVqBBnjf2vorRIKqvUezR1k5ZrOttBKEdZbixLK7amqSDWU8ngDfbSuzhHWW/Ip1hIKcst5mldm5RD2+kyRDmMIY7Qvo0adUgFch2C2Jj0FYUQSJSgtMfXG+avEbR8SjU5iFIo4rSjCXccxU6lpqVqyOgqlayVrYz/g5U/8CfmRfna97RgjjxylcudedCxCc3EQYnsug8cOkRuu0LgwiT/fkHDYUm5yYth8s24QnURSumjZaeNTavDiEOZfh76DUiHkVyUX4M+vEBFJg4+WJ3XvsoOtexEbwkrjqT3nq5QkxYujUstfGxcHmARph3SrK32hlGgk7X4L5CpmpX+L6E3wOn0FFNIpmxuu4PQVOkacow7BAAAgAElEQVTdGyhLHj8jlXEAUFozcvweZr9zphPbjpoS0mlPVWlcnEQnSZrblRBNEidETZ84CCEWbZ44Wues7M45gD893/Ni0rv9JTO0lUK8V9dp9W6/HpIwJpirSbLZspZVSC2tfmrPLFC5c++ibXpX9PmR/k4YqD27wPyp87Qn5xh72/2dbmJ5MTdWjbVezLftRmlMSKx/4GgqEdHqPpbEsOsR1ODR7oSwOIT6xXS7rAvWlkanTP4gSWT1nDVBdcjCRrZ8kG9501hqCZK08iJLXts5yXMURuVuNw/5QahdSEM9bvfpTkHyAG4Ja+jeW/Iqbnd6Y9Fho0315DlQMHDsEG4xLw4hrY+vHNnP5DdPErWDNGwjapzBXB2nUmT/+95KfrSf059+ktrZCUg04UKT1tU5lGuj2+lnNk4/+2FMbEfoKCE3LM1erSBcNrVrU8iuTOIb77OxCx46jpl65hX2PvHoogqpy195jtOffpIkjMkN9tGcmKF5eZr8cGWxMQdQitbELK/93pc6CfPigVG5YmoGTD51ir1PPLqosmjdmks3gHEAN0hHJ8jJi0RxWJf4tu2l4nCWDDYp7YHxb0hJqFuRRHESIKETRwwkFiQ58BfkqiHSIoWcrah1Ir+VBU4ZWhPc2kSxTq98XKnc8WeRfIcrvzPp67idjsPUkHZEyurfloqpYD5tmDNsNb0yDcX9o1z+8rOdK4DGxUn2vvtRwvlm+inTtK/MMvzIEaaefonID/EGSjQvTWM5Noc++H1863/6BMqxcPoKBHN1lGPhDpZpz8xLd66mW32TotuyUvfn6nh9RUngblOSdoDKe/iT87QuTdO6Mkvshzz3v/+25DpsG6dcoD1Vpe+uvTQvTXcmkoEIykUtX+YeTM8vagazbJuxx+5n8ulT+FPzzL7wOvmhvuvSXLpRjAO4QTo6QSCG2eu5TGvPdmrWde0iLJwTqQgvFYFrTohR15F0DbtFKN0B1mVZPetEcgpBLb0a0LJqtvLgzyBx9Vt8FZAEQAjNALySXA0FNWnqyq6G7Dy0prpffMsW45/rF0eobAkVGbac3gRvc2KGuB3gDchqNajWpcRyzzDhQoM7e2QR7IJH7fUJqqfOM/vCa+QqZc7/0dfRie7UxZcP7mLhzCXCepMkjEXfP63rX5EwJpitrfLgFtNpV0kbGeMYlXOkPDXt9kUpnvuF32H2hdcI5htEtZaMrwwjsBS1M5dwygWCaoPa6xPUXr9M3JLmNgWc/89/g+UtNr1ef5m9736U2RdeY+zt97Pv+4+vW3NpI5gk8I2S6QSFS9rKl9as18ZTraD0P9J2pQ7eztNpeMmGxNz5gTSu7ou0hFOQFX//XTDyPfKczoStNPl6S7HS12HJ1Y1TgPa8vJ7CGBSGu/drLds6xdT4IzH/FRq+DDef3gRvFq/v0DM/F6U6gme54QpxK6D/6AGiRpv+IwdwB0pYWTioWmf8C9/m0l88Iw1X802SJEGhUdt0ItYilJKSz7wnIa6ih5VzsRwbC0XiB9QvXsFyHK58/bsE1QZuXxEr52LnXEgS/JkFolZA5chelILJv/ku4bzYCK+vwNDDdxNUG0yfOE0cBIsOb9k2+aEK+77/eEeD6WZjrgBukNWbwwrXVqu0PamIaV6RVbHtSTz99c+nzVBOZ/AJow/JFUJzSo6jw9X3K2fGxsJDizJ969i/Br8GKg1TkUhYx49lpW97aUWUFnloy0urotJpYaa565bQKwPhFPOLH+ydbqU1cRhx4uc/1albb8/MUztzmX1/77gkeS/PyJB2IKg2cMp5Kkf2kRsZIGr50t2bbN/wTget0UFEfs8w4XxDRlEmGlv3FG9HMZNPnUTHCU4pL4nw9H20XIfYFxE6ZduUD+8hrLco37ELHccsvHqZ6slz6EQTzC5w7j99gwPvfWsnT7AVMf+lGAewAVR+EO56/6LmsGWVLH0HxKDHYfcqQGspgVQ29B+W8NHsyxIOalyBgTtTuei2lJgOHJFEqs6Srr30GmOH6wsN9RrzVOpZuWkj12pf2J7j2V4qgdGSK5OsES4bBO/Py4q/vF9mJGdT0FJlVSPrfOvolYHIjw5g5z3Rr0eGs3hDfVRfvkjUavPaZ79EbrjSiVkncYwGJr95krDRRmmw857o96Qfz8b4FPnhCrve+RBXvvo8sR+sfjLbBdtCa40/WcUu5YjnJfwaR+nnWgOORWtqDqVs7IKHU8qjXJskiqVBTmuSOCYJRDE0N1KhfGAXl7/8LDK7QIy9jmNiP+DSX55g+PhRLGVtScx/KZs1E/g9wG8icYlPa61/ZcnjHwJ+HbiU3vVbWutPb8axbzVLm8OWPd63H91/SAx49kGKAzGy+SEx/u0ZWd0rJVcC1ddldQxiLFuTPcY1GzeZfkg683Y3GM1TVvoB9yQJvaaKJ4CdSl9n08wcOk4jm3UcBeBoCQUpBcP3gVs2JZ/bgF4ZiOb4FKUDY50qoPLh3Vz802+CgsLeYeZPniM30s/Y2+7HKReIGm2SIKR1tY1ybKy8RxJGUuqpwPKk6UvHmuE33UVp7zBn/v2XSXx/ewveJglaWVh5l3ChKdVLWdmoSvt44kQcRRLTvjqHU8pj5z3ak9WOlIZTzFMY62ffe97MuT/8605PRGb8Qa4Wxt52jNbVOcbeeh+jj92/JTH/pWz4G6iUsoFPAE8A48AzSqnPa61PLdn0D7TWP7PR4213lspFUNqDOvQe9MWviqGPI7DaYnAH7pYn1caRyppU+CmbIuZXZcUctZFvlpV+gdLpYyprxso6jLP8wFKyQS6Zm8q+hT0NaZkhTyuYUDk5dthAupPT4yi32+SGku2zOQBRU/adxOm2Wnogoja4JdTom4zR30Ys7Uh1K0XiIOTUx/8Tgw/dSeXufdTPX5VRiFoz8dcvSJij6ePPLhA1fZRjU7l7H+3peeJmmySMif0QZVlUjnb18PO7BohrLfzZhe2rdJJ2IsctXzqFNfKdi9PvVWeGkYVbLhC3A+ZfuiAaRaiO6uiRn3wPe9/9qLwnYUh7ZnGCO25Jd3BhzxA6Tqgc2X/LhsZvxrfxLcAZrfXrAEqp3wc+ACx1AG94dHtOcgJpFYyGbqjjrg+IY/AXJBxSPSOr97hNR0MnQ6V5AafYbaIiFZOzNChPjGuSJYOz5hbV89O71MqE2DxxHEkgnblJm67TSJ+XBGl/gidXKbaX1m4nacNbTvIQXp+8TlGvksdtT2L6wUJa9ZFeSaQdvsb4b0N6unclng1epdQJ9zgFKeO18x71sxN4gxWiRksMu9Yk7YDqS+c7K1c778nnUsntJI6ZeeE1/Ktz8oncrsY/I9GiVmrbMsMj61tIwz+Z8S8f2s3sC69J4jjvyXrMVrilIif/nz+m+tIFbNclCSIWXr0oncTpFYJd8Doqoje70etabMY3ch/Qq286Drx1he3+oVLqncBp4Oe01itrou5QdBKJ8de6O0QeIGyiL34Nddf7Re/myjOyqvbn0+7gdCWfDZe3rO5gGTSU9sLBvwsXvgrNq7IyJ0y3Vek8Yae7uO98oXtW/cqWVXt2heEWobQb5s/2rPhVt7FLR+IXnIIondpeOoxdQXFMjH8SS/mnTkTsTiey3/475RitadnPgcdRffuN8d+GrDSaMFhoiBplSn5sALvg4c8ukIQxrauzEvaxFO5AmXChgQ4iYg25wXJHoyeJEq5+42+Z+PoLhHN1KZHcxnX+i9CgbAs770o1VOoELEcq77zBMu2pKgBOwZOviueApaRbOIrxZxYYe+v9AORGKlz52guUDo5S3D1Mfmxgyxq9rsVmlIGuFP5e6uf/DDiktf4e4MvA7666M6U+opQ6oZQ6MTU1tdpm24/GhKyIe+cFgPwdtdC18a6DKIzA0L3SRKYjaYiy8/IltPNyVRC1AAX73oE1cBfq6A9BeR8EsxIaihpi/JUls4eXGf5EwjUqrbbJxNfsHBR3iVhdEoOTo1OOars9/5uqq8ip0vCTU5Du3j1vgcJQN09Q2i37zTR9/Cp4fai7/wFW/yFj/LchS+f1lu/YRfnQbizPo3ryHEksIcGsSSluBYT1FlGzTdz0Cett4nobO58DS6WjMJKOnk1+pII/VyPO5B52ivEHWVil8wd6Be50lJCEEe2peYn5JwlRy8d2bCzPSRveNFpB9eT5zntY3DXM4IOHccsFEj+kOT5N/dwVlK22POm7lM34Zo4DB3r+3g9c7t1Aaz3T8+engF9dbWda608CnwSRg96E89sSOp3Bq1EbF6OeXR04BanuaV6B+XPSEDV8TIx/HJAJyanBNDbo9YlG/uBReVyOmnbdNulYbmXRGb6iY3lepq/uFiTv4JakIqfekiR0Nrc3iaVjOWqwyK9nej9OIVX8tOWKplcauzBqFDx3EKvNA67cvZe5v32NhTOXGLjnIIAMU7cUludgeU4nLBQsNElavujmey5uXxFvsEzj4hQ60diuSxREayt8bkcSjbIsKV/VWhyC1p2xlFG9ld4HaAgXGsSuKwY/lZvQSdJppgNwy0UO/fB/Q2HX4IoS0rBYT2gnDYR5BjiilDqMVPl8EPjvezdQSu3RWk+kf74feGkTjrutWNQZvCJLHo3a3RkCtitXATOn0tW0J0nT3rh5pj1U2i3OIaxDuypXEFpDfkBW3ovGTqbJ2SQALy+Oo3pGHE1xDBqpn3byaSgqEEmK2JfnZ+JcSkFuQMI7URPllpZVPylYsxrKsHWsx5CsNg/YcmwGjh0i8cPOkJj2zDyWY1M+MCbdwJ5LUGuSBJEYPCURQ39mniQISYJQrgbiWDTYtvHMkdWQJLAG28LKlE+VQjkWOk5QSsk3Ohsik+Xh0vxH1PRFIbWzQ01h1+Cqyd6VwnFbMRBmww5Aax0ppX4G+CKSkfxtrfVJpdS/BE5orT8P/M9Kqfcj2cpZ4EMbPe62o7czeOnYSKcg/QC1C3KfTsT4a+QxgPKBdOUfwt53LIubd64weh2H1qnmTrpSyfYV++mQeSUNZKkSIpYjDTlT34HBe2UiV+TLccM6nVW/W0lDPiV5LbmKXClE7RuazLViZZS5OrgprNeQrDUP2C3lueenfqAj/7Dw6iWmnn4Jd6DEuT/6a4L5BmGt2V3Za9BhTBQnRA2pWNNaQ6xJSDrnsZPoiNIlMYmTKt9a6euyLZxSgSSKiLNBNqkjFGV0h6jeonl5mr479xDON9eM9S8Nx2X4szVOfvxz1z0Q5nrYlG+h1vpJ4Mkl9/1Cz+1/DvzzzTjWduVancF4fejMQeio0wwlyVxHjKyyREfIspcZSLnC0MsdB1qSw0mQNmXR1dtXSpyNTreLAzlGUIPpvxXdoWBBktGQ5iBcGLxbvrCNy91Er1+9oeat1SqjMBO+Np3rMSTXmgc8/PCRzra54QrT336Z/FCFO/7B3+H1//Dl9LOiugqbixrENXEzkMcTvfNCQEvpzA8WR6Y8VwbSN3ul0JGZBhp02u8z+92ztCerjD52Pw//4o+vasRXC8dtxUAYswzbRK7ZGZw5iNZMGmZBjHbl0KIyUB02lodSSnskvBO1Ja6fYbmpDHMsCeRMoz9JJSMyJ9C7RzsnjqdxGfrv7u5H2bJ984oonJJIsnqdzVvLRmYWRmGNyijuer+5EthErseQrGcecEavs9BRTG64gnJskiAkavrpRyv9nGWfYwW260j5407AsaTaJxNeXOqzekJZOoyJw4iklb62rEmsN9FtSTit7669uKU8hd1DrMZq4bhs32YgzA5irc7gzEHoqe/AxLdFD8grL+4BgI6S6NL96qH7oXZp8ewByxEjPv8q0oSVdEtLscURKIvOOErd8yFXtjgV2+u5okD2H6QVRm55XXr9K670k0iMQmmxQcItylVSY0JCY4ZN4XoNydJGsJUSk7DYWdTOXSEOI5kNnPOwlZJuWaUgG/KeNk/FTX9nyE0q1SlfBa7dq5AkIvCWbq9ce4X5wgqvUqI9VcUbKK+5il8rHGcGwrzBUJYDo29CL1xIDXHPN2SpkujS55Z2o/sOpBPEAvntpe3lrUlJBGOLBLNTlAqhrIFMWXLbq0gVUNhgTbmHRMJFKzmjpXR6IJJYHFI2FyFoQHtaEs5quSVY8UrHcMPciCFZbR7wUjJncfZzf82rv/MXtK/OkcQJ7atzxHGCcmyiRkvCl45FgkoXAOkOnLSxajtGg1IRuHVjK5StUJaUTSe9E8iyhnnbImq0cQoeQbW+5ir+WuG4m9knsBP88xsOZTmSF1BKVsLZz7U6Zkt7xHhbjvQSZHmDqC1hpOIuidkrS8pJ80PgDcjftidJ3Vw6gMbOSQdvrpIa7SUqo0my/oRvY0LyCPWL0hTWmJDf/qxcgWTKpkvfh3U4F8P66TUkvaxlSGI/YPq501z60gmmnzu9pmib7bkc/ofvZOiBw4w8ehS3mJdqnzBOJaUVTqVAEieQaehkBj/apsb/erEtbM8jN9iHlXclKdxL1nSfc/EX6rSmqoQLDVpT1VXf4+wKS9mK+rkr1M9f3bI+AbXsBWwjjh8/rk+cOHGrT+OmcSPVMUtDLcCiRHNnBOXMydQRpKWfIKt/KxVxy49KrL9ySP7OKosy0bnBo6iDjy9L1K50znr2FXjtTyUHYfd8WKNArgD674S+/d37w6Y4O5MD2HSup5zwRksP6xeu8vwvf5app07iVxsE83UpCU0SlON0BdTeSGR5DgWFPUPseudDXPrCt0iiWPIccbLcwSlAKSzHZve73kRx9/Ca73EchNcMx63rVJV6Vmt9fF3bGgew81iP4+jdhjiAyecl7GM5aUdvCcYegcnnxJloDXFT8gJ7HkMN3rN8n6s4HwrDMP516RVYSntGBuD0Jq6zaiJTBXRTWI8hif2AEz//KXSsl4UdlK0WVQwt7Suo3L2X5/6336E1OY/l2VieS+vKDBNf+w6WbaM8h3C7TPjaDJYkhZ2hPtxCTiSd2yFhtd7pHu5IcGXPsS3cvgJ2zuPuD/89HM9b8T3e1NO9Dgdgll87kGtJUK+0jR6+f0Wnofv2r+sqZC2tI+ZOi+PonXkAaYmrB3vejioMmj6ALWI9cf31VgytdJUQBxFJEC4LKbmlPB3RTCuVD8ni/irrTk83XqnSZrvSe56ORdJo0ZqvQ6xRnpRe28W8NMAFPaEvpXBLBbxKiaDWZOGVcYYevHNLyjvXi/kW3ias5jTW40yArtZRfkk5m1uUOcVev4yyXFqhVNqNKgyiTJfwtmI9FUOr9RXMvHCGhdPj9N93sCP3nPgB+dGBjkhaVjapHButNUopUEoMpGNLTmAnYKmu7IMlfQ1JGHd6GzKDHzfb0geQhX1ckc2gWxVLsNAjF3OTyzvXi0kCG9bFmlpHdlFE5coHoO8OSRz33SF/5wbM2MdtyHoqhmaeP8PCa5fxqzWaEzMdcbPC2CCJH4ogWopTlPGIhT3DjD12P06lgLLtjsS0jjU665aNY3bKakB5DrbrohwbpRQqdQLLSHSnLNTyXFH8dBxUOipTIzLbHW6xDHSGuQIwrIs1tY6Ugr1vl7GWUdptrCOwzdjH7cq1Sg+9Soln/td/S/XUedy+Ir069vmxAZTnUH1lnKjZxinm8Qb7JLGvYOihuyiMDnDhz76JP1vvzgNOZwTE7XBnSEMoKO4ZJm75JEFE5IckzXZnlb/MEWhAyVhJHcXYBRdQhPUWtudSuUcKIbaDDHSG+WYa1sc1tI7U4FFRKjWaPzuCtTqB7/0fP8BLn/gTLM/DLRfw+mXlGrcCJp8+xcib70VHMc0LV2mOT6LSkMfgg4dxygWZIja7gMq52DkXp1LELeeJmr5oBQUhCguVc3FyrhjWdrD9JCOUorR/lMb5q0StoNsotpLxlwfwRvqI5hoEC03yIxXJC1sWw4/cTXtibtVu61uF+XYa1sW1tI46ht7E+ncMq3UCz714lmCuTuXuvdTPTRC1fJxCDrvgEczVuPTFZ3BKBfb9/TcTzNSIWj5JGJEfHeDef/p+Xvi/fo/29AKEMVbOwXIs8iP9WJ5Le6pK/fxVFIrivlG0TmBmgTCKr68ZaytQEMzVsIs54qtz3fLWpcY/DQspz8Et5Bl+092S7B2uUDo4xr73vJnGuasbLu+8GRgHYFg319Q6Muw4VqoYyhLElmMz9rb7mXzqFEFVmvnac3WUUhx471txPA8n1bsHqL12mRf+z39PfmQAZVmLEsKN8Sn67txLfnSA1tU5wlqL5pUZnEJu2+oFuf1llG1jWxYD9x6gNVmlPTknDqBTzZQmiG2F11fE6y9iuy53/ujfXfS+Fob6b8lruBbmm2u4LtZdNWTYsfQmiL3+MnufeJT2ZJWo5bPwyji50Qpef3nZ8/z5BrEf0H90P7EfYNk2cepI4nYg0gjFvAhn5lwsy0KHkWjp+OmIU7h5oaClpahr4dgMPniY8oExnEKO/NgAsR9w5rNfJqw3JWSV5TEU2J5H3+HdhLUWlmdvi/j+ejAOwGAwLGJpgtiybYp7hvFna/TdvQdl2ys+L2752AUZPJMfHcAbKBEuNDrDVJIw6shUDNx3kMEHDtO6Okdrao7aq5fRSYJTytOekfDRenD6iyR+mMpqKRGjC+NUrtpaJFetLIXWPRLWK5H2L9iuQzCzQPl7H8SybYL5OpNPn8LJu0CBSINGo1Aoz6G4Z0gmiCk4+pPv3TYhnmthykANBsMi1tKmefiXPkR+uLKi3lB+rB9vQK4MLMdm1zseoLBnmCSMCBtt/LkaUa1BbrifXe94gPLBXQw/coSo1kI5Nk4hh1su4OS87tXASijAtcG2iJs+3nAF23NEgyjV7le26lQdOf1SxaSjZLnxtyywFVbBw8q5sn0hh3JlAHx7skoSx1z95knCWhtl24y8+V7Kh3dT3DVM+fBuDv1338vI8Xvov+8gex5/hL3vfnRz/0NuIuYKwGAwLGMtqejVqoce/qUP8fK/+tPOlYPXX+bA+x5j/uVx/NkqB3/wHXj9ZSa+9FwnhNSerKJjjeXYnZCK5TrdcM1S0q7buB2gbBnPGMzUJCSTYVtg21jp9C63r0hUa3V7DzoDXpASVaXQSYJlWTiFnDS7D1dIopio5VN7fYL66xPS0GYpGuevSqPXgEO00KQ9PU9+qEJ+99C2qe5ZL8YBGAyGFVlNUuJ6nUNp/zBv+b//B8oHdxH7AbPPn8GfreH2F2mMT5HEMUkYoRwbt1wkmG90Z+5mZDLLrkMSxihL4RTzokKqwMq7JH6IckWMTocJeC5Ka/zpeZxCDivvEtVbKNcFxyJutlEotGvj5HMU9g5hYWGXcoy+9T4mvvwc/swCcy+eJQ5j3L4CpX0j2HmPqOWDhv57DjL29mPs+/7j26q6Z70YB2AwGK6bG3EO0A0vPf/Ln+XyV54lqDUJqw0s14ZEE7Wl6UrCONIroINIavDt7kB2p1LsOAnbsojaAcqRiXZWzkUnGiefI4likaxGQ6Dk+a6VKpem2ytIgojEj/BGKow9dj861ux99yMMHz9Ke3YBO+dR2D3Y6ex1CjmpjLIU+77/+C3X9LlRNsUBKKXeA/wmMhT+01rrX1nyeA74LPAoMAP8iNb63GYc22AwbC+uJUZX2DWIW8oz+OBdaDQzJ14RnX3HAaXpv+8Oqi++jlYWbrmA1glxo41d8PBnajKEpR1IcjmMiRKJ7cvk00QcRaKJWz46SdBRjI4tlGV3wj1OIY9OfCzHxu0rElTrRPUWxYfuJJirp81a/4j5Vy7Sf3Q/C6fHSfyok+QGcRo7qeJnJTbsAJRSNvAJ4AlgHHhGKfV5rfWpns1+EpjTWt+tlPog8KvAj2z02AaDYecxd/Ic4UKT4p5BJp86hZ1zaU1WxVjHCeVDNsp1yPWXURZSiz86QP3CVdmBUh25BUBm+UJHgE7rRAbSWwod63ROtkbZFpa2QUFYa6AcB7dcpO+uvYTVOvndQ5Bo7vmpH2D44SPYnkt7eh5LWYw9dj+TT6f9EGloa6dV/KzEZlwBvAU4o7V+HUAp9fvAB4BeB/AB4JfS258DfksppfR2HkZgMBg2ndgPmHr6JRrj00w/exor75IfHSA30k/UaNOanMOfqVHaN4JTyHdW3O3peUg0Tikv4yfrLTHs2Yxri9QBIH8nGp1IctmtlMRwJwk6jtGxiNO5lRylg6Mk7RC7mGP0LffSvDiF5Todo56VxEpIqNsPkXU+X6viZ+kshcFjh7Bz3prP2Uo2wwHsAy72/D0OvHW1bbTWkVJqHhgGppfuTCn1EeAjAAcPHtyE0zMYDNuBbLbAwmuXqZ6+0GkMs/bZ2HnRHQoXmkRtn8EH76T2+uXOijtq+aAUxd1DaK2p10V2XFkKHck60i7liettsCxyI324fUXChWYnzDPy6BGSMGLu5HmiWhOvv0TcCjoid5ZtL5Np7tVMal6ckpyBZa2r4udGJ65tJZvhAFYq2F1pONq1tpE7tf4k8EmQiWAbOzWDwbAd6J0tMPTQXSy8eomkFUCS0Lg0Td/h3SR+hLIVtuegbGvRijtqtJl94TUGjt1B9dQF7JxL7IcyZ8C2Ke0dRtkWzSuzKKXwKmVpPotiLM+hMDZA+Y7dFPcMUz68l0t/+Qz99x6ktH9UpJuz5rYVZJqvldi+1uvtnaXgz9Y4+fHP3bRpYNfLZjSCjQMHev7eD1xebRullAP0A7MYDIbbgmwCWdZZPPzIEVCg44So2aZ1ZQ4sGH7kCAolZZtpB3Llzr0M3HcHtudieS67vvcBCmODuH1FnFKO/HCF4r4RCruHsR2Hwq4hdr/zexg5fpTS3hFyg324lSL5MRlZaudc8iP90sy1Z7hj/NeSac4S2/uekIqfaxnv3tfbS26oj2CuztyLZzfpnd0Ym+EAngGOKKUOK6U84IPA55ds83ngJ9LbPwR81cT/DYbbh6UTyPru3EP58B5yo/3kBu+sfQoAACAASURBVMtUjuxj77sfxRvokxJPd7HcRLjQZOxt95MfHSD2QyzPITdQxnJdvEqJcKGJPz1PYe8wY+84Jnr8nkv/PQdQjk1p/yjN8Wnq565gew5v/8TPYrvOsk7nzWrkWs/Ete3AhkNAaUz/Z4AvImWgv621PqmU+pfACa3154F/B/yeUuoMsvL/4EaPazAYdg5LJ5BZts2utx9j8ulTJK0AZds0L07hDZZ5+yd+lrN/+FfLOo3f9Is/TmH3EHMvnqX60nnGn/wWSiuCeou47ZMf7eeRX/4wpYNji8I1lbSMc2n4ZvjRo2uGdTaSwF3PxLXtgNrOC/Hjx4/rEydO3OrTMBgMGyT2A078/KfQsV4UFmlNVwlm69z1jx+nsHuoY4TjILxmzH0929woG03grvZ6/dkaylY3NQeglHpWa318XdsaB2AwGLaCnVAVA5tnvG/V670eB2CkIAwGw6r4vs/JkyeZnp5mZGSEY8eOkcvlbmhfN1JNsxY3q8Y+S+D2Vu+AJHDr564w9+LZdUk/bPbrvRkYB2AwGFbkwoULfOxjH6NarYrujtYMDAzw0Y9+9IZ7dK4lE7FebubqejMTuJv1em8WZh6AwWBYhu/7fOxjHyNJEg4dOsQdd9zBoUOHSJKEj33sYwTBrRvjuLTGvnzHLsqHdqNjzcmPf444CDe0/52SwN0MjAMwGAzLOHnyJNVqlaGhoUX3Dw0NUa1WefHFF2/Rmd38GvveiWi9rNUnsFMxDsBgMCxjenpaumxXQCnFzMzMTTt27AdMP3eaS186wfRzp4n9xVcbN7vGfq2JaDtt4Mu1MDkAg8GwjJGREVarENRaMzw8fFOOu57Y/laEaHZCAnczMA7AYDAs49ixYwwMDDA7O7soDDQ7O8vAwAAPPPDAph9zvfo5S4fW9263mSGa7Z7A3QxMCMhgMCwjl8vx0Y9+FMuyOHfuHOfPn+fcuXNYlsVHP/pRPG/zJY3XG9u/nUI0NxtzBWAwGFbk4MGD/Nqv/RovvvgiMzMzDA8P88ADD9wU4w/XF9u/XUI0NxvjAAwGw6p4nscjjzyyJce63tj+7RCiudmYEJDBYNgW3E7ll9sF4wAMBsO2wMT2tx4TAjIYDNsGE9vfWowDMBgM2woT2986TAjIYDAYblOMAzAYDIbbFOMADAaD4TZlQw5AKTWklPqSUurV9PfgKtvFSqnvpD9LB8YbDAaD4Raw0SuAnwe+orU+Anwl/XslWlrrN6U/79/gMQ0Gg8GwCWzUAXwA+N309u8CP7jB/RkMBoNhi9ioA9iltZ4ASH+PrbJdXil1Qin1tFLKOAmDwWDYBlyzD0Ap9WVg9woP/YvrOM5BrfVlpdSdwFeVUt/VWr+2yvE+AnwEuOG5owaDwWC4Ntd0AFrrd6/2mFLqqlJqj9Z6Qim1B5hcZR+X09+vK6X+CngYWNEBaK0/CXwS4Pjx46soQxkMBoNho2w0BPR54CfS2z8B/OnSDZRSg0qpXHp7BHgHcGqDxzUYDAbDBtmoA/gV4Aml1KvAE+nfKKWOK6U+nW5zH3BCKfUC8DXgV7TWxgEYDAbDLWZDWkBa6xng8RXuPwH8k/T2N4EHN3Icg8FgMGw+phPYYDAYblOMAzAYDIbbFCMHbTBsIYmOaYRVwqSNa+UpuQNYyr7Vp2W4TTEOwGDYItpRnYuNU0RJ0LnPsTwOlO4n75Rv4ZkZbldMCMhg2AISHXOxcQqtNXm73PnRWnPx/2/v3GIsy87C/P1r7b3PrW7dPd0zPZf2jAcb8CVc3DEGogiCQ6wRwYEEibzEBCILRbxjyVIU5SUhPCWBCBwUiUiES5AmOGGIjUMQLxg8vuHx2GOP22NPT8+lp6urq+pc9mWtPw9r71OnLqfqdFfXfX2tozrn7H32/mt1nfWv9V/7z+PVHbWIkTNIVACRyCGwVt5iVK1R+ZzCDVA8AKlpUfmCfrlyxBJGziLRBBSJHDCjap1vrz1Hv1rBSuhtayrLQnqRxGQAlD7f8bPRZxA5SKICiEQOkMoXXFv9HKXLEQQrKYLgtGK1vMm51mUg7AS2En0GkYMmKoBI5IAYVet8Y/WzrJVvYkgofUHlC9p2HisJlRYMy1WypEsvXdr02a0+g4bS57zcf54nF96z607gsHYOcYdysokKIBI5AJoJ3PkKS0piMqwkDN06Q7dGZjp4KhTPY713bJs0++VKrSw2r/RT02Lk1umXK8xnF3a892HtHOIO5eQTncCRyAHQTOCZ7YzfM2Lp2gUSk9EyXbrJEo/NvXs8WXp1rBW3WB69wp3iDVSnF8PdzWdwt9FGk/ddK27NFJEUo5pOB3EHEIkcAKUfAZCZNkYsTiusJIgIlgQRSyeZYz47D2xfTVc+Z1CtkZrW2FE8yU4+A6+Om8NvMShXwmSMR+o13rSdw72u4vezQ4kcH6ICiEQOgNS0AaX0IzLbYVitouoAwVFixXI+u8xK/hrWpLw+uAbIeEJV22Xk+qwUr3Gh9SgiYSIvfU5isrHPoLHB98sVlvNXKNyQkV+n8KNtkUbN5xv242doFNw0pu1QIseLqAAikQPASsJasYzTEgBBELG0TIeW6ZHajNeHL1FpQemGFD7nQuuR8ecFw1L2EMv5Ddar2+MVf7M6N2LHq/fS5ayVtwBFxCBYEsk2RRpN7gQadl/Fr3Fz+C1S09rRuRsU3HR22qFEjh9RAUQi9xmvjm+tP4eiOK1qW76CgKqymFykdBX9ahnnSzyeSgveGH2ThfQSme2QmTaJyZjPLrCQXqRt53BaYCUNuwCfjVfvRiwGQ2IyKl9S6gCrFiMJhR+yViyTSEJmO5uijaat4itfsFrcIq+GtJIusN0s1EuXSExG6fNNk/3WHUrkeBMVQCRyl+wV+ngnf4OV/DUEoWW7BF+uoiheK4blGkO3isdjMHh1KJ5KHXfK12lVPaxJWMguIggdO8dycYPKF0Gp+ILKlyDCYnqR0YS5JTEp3rdwvmSkfbxWOF+RmIwFk1K44XgS32kVr3hWy5uA0k7maNmgALaahYxYHuu9g5f7zzNy6xP3z3aMaoocT6ICiETugr2cpqNqnZfWvkjph1hJca5ERGiZHkYsg+oO/WoFIUzWiqJaASHiR1Ux1qAoK8VrLKQXWc5fQYFEMlbLm3h1lG5ERUlRDchsE1IakswEwRMmeK8V8+l5uskSlZa83H+eJ+a/l2G1Ru4GeHUUbjiOVirdiMoXJJKRTSiInZy77WSOJxfeUyvDsBOIeQAni6gAIpEZ2ctp+sT894bjgIgZT4SqnpHvk0obrw6PG5eEUPWAYkjqSVzHZhSvnl6yyFq1TMt2WR69ss2JW+iAwo8QCaablp2j0gJVJa0n8W66hGBIpUW/vM0LK3+5SbbV6iYdP481KXnVx6snsx0KPyIz7bEDuvldJzFiY7TPCWZfeQAi8jMi8mUR8SJydZfzPiAiL4jIiyLykf3cMxI5aKbFxTdO08SkFG7AsFqlcAMSk1L5gluj61S+oJvMIxi8+vE1K1+QuzV8XQTOaYmqR9mI9RdM+CcJloSW7VH6EkW5k7/OenWb3A8o/JBCh1B/2oghMx0UpV8tU/oRTgsKP8Ljcb4CgnmnX93B+Yq2Dead1LRomS5OKxbSS+PJfuTWWSve5Hb+6qbdTmPvv5fcgcjxY787gOeAnwZ+c9oJImKBXyc0jb8OfEZEPh4bw0eOE5PhlLfy6wgGEQE2TDylH+F8xe3q1U0TnqksLdNjVPVRFEFIametV3A0E7ANNnWF3A8oNcdIUn8mTOeOitz1yRkAyqhaxamj1BFjM9GE0vBUwb/sbNh5YOkmC1RahmxjdeNIoNKNUByZ7QRHb21OgqCgXu4/R9fOY8Tg8aSmFT5f3GQuvTB27sYM4NPDfpvCfwUYf1Gm8F7gRVW9Vp/7e8AHgagAIseCjQktZ7UI4ZSJpHTsPIhQVANeXv8yFztvqSt6JiSyEVvvtKJfrbCQXWStuIURg9JM/Juzea1aCs1J6zBNg4V64gdISDFigxkHz8ivQ60epmFJyJIOhWuUTzv4AdSNaw6VbkTuhgiW1GSsFK+jquPfo6IkdwMqX5BKi1JHtZO7VTuoc57svQdgkxlM8ZRuxLBc49rq53j70vt2TFyLHE8OwwfwCPDyxOvrwA8cwn0jkT2ZtOsbLEYMBsvIrTF062Smg4iwXq3QNr1drqSs5jfD6tk7ct9n66Ttqej7zXX/E8kw0sL7kCSmQKl5vSswgNt2na04HKXLcVqBKLlfw0j4ale+wFEycushYkkWqXyBV0ciWR2qWlL4Qe0+ps5X6FK6HK+Orl3kwc6TtJM51opb49yBrbuIgSt5YeUveXLhPXEncELY0wcgIp8Sked2eHxwxnvstD2Y+hctIh8WkWdF5NmbN2/OeItI5N5o7PqpaYWYfbSevA2CYMTUq2Tl5uhbdJNFBKF0ObnrM6juUPoRhrDSXkgv1qv23Sfthtz3MZJgSBAEpfELBIPQLHjK+n4KKggWr47KVbSTHi0zx0Pd7+DtS++jlfTI3TB8Th0jt8aoWg8hqnhKzRm5ECGkdX7C0N0Zf4mb3IEmXLTZRSSSYUlxvoq1gE4Qe+4AVPX9+7zHdeCxidePAjd2ud/HgI8BXL16dbZvUSRyj0wmQ1lJUHXj5CqvDu891oLBImJQPPPpBZaLGxQufNZpRcEQ45Lg3MWzl9lmA8Fp7Vj2g7GV/27JtQ+AIWHk1oJzGGG98BhjceoYVms80vsuXlr9AgNXkvt+uJMAutmv0EQRBaUkLOevcK59eZw7ULrReBcxSeNfiLWATgaHUQ30M8DbROQJEcmAnwU+fgj3jUT2ZDIZKrVtqFfhDcaYYKsXS2Z7qNdxzZ3aYDJetTtK1qrl+pOzTf7BVu9Bw7XuZfJv7peQoTg8HsUFh7IOKN2Qb65+nmurn+f6+vNcmXs3HbuAYGnbHjsVHVUchY5qW38vRAWVt8YZwM0uomE8Ro2CiLWATgT7DQP9KRG5Dvwg8Mci8on6/YdF5BkADVkuvwR8AvgK8Aeq+uX9iR2J3B8mSxoIhvn0AiC4OvzSe48gLGQXUXV4SnI/xFPhKPGUE1U3FSaUx96E7GAjhsy2Mft0yTma3cfkjK5UlJQ6ZK14k5XidV7uP88D7SsYMeRV+F2mydeYhfrVCt9e/RK3R68yn1wI5iFfUtVNbpoxasJIYy2gk8F+o4CeBp7e4f0bwFMTr58BntnPvSKRWZhWpmHa+9tKGojUkS9KL1kiNW0y06b0OUO3Rst0arOIqdfrwdwzq71+OwoKViyZ6VB6oeLeVs+6h/KpyPFVRelGjMqmfENQGE2xuK3XyEybVFoUqvTdHV5cfZb59AKpaSEiJKZNJ5kbJ4zFWkAni5gJHDk1TItPv9R+nDdGL02NW99a0uDB9uPcym/gtMTj6Fe3cb4klY2sWFNH7Oxktmmyemel1BGDSnCaIxxsGQWPx+uIvlshM20cIfqoidUQzDiXIbwWnJZUmpNJFwj+kZbtIhjWq2W8dsj9AIi1gE4aUQFETgXTyjQUfsjX7nyahfTSrjXvt5Y0ONd+mNujG9wYfD0YdhRGfgUtJ5OxdrLZC4nJcN7gKJgFxVPUTlzuQnHcG03NIU/u++NchWYHYwklKiYVWHD2tsYZz17DsXYyB6Kcbz1CatqxFtAJJLaEjJwKJsM5N6E6EZmzQWpa42iVaSwXr9KyPTp2nkpDFrAn1PJpQja3YknwvpqYQHdNkjwyHOU4EU0mpgGPD53LCNnNi9lDdJMlVH3IbKbaMsELqWmz1HoQgJX8tVga4gQRdwCRU8G02vauXq02q9btn9vZ3t4olJbpcjt/FSM2hImidflm3XH9v1Gm4fhHMIeJf8P8AyH6x6nHiGUuPY81lpXRa/X4CkaE9fI2VtJxxq+q58U7n2Hk+njvMCZEF12Ze1dMCDvmRAUQORVM61Bl64zYJjN2++d2jlZpFErh63h3k5EaZehWx4lam5GJXYEdKwKDvSt/wGESwle3Rg0JIkLbzLFW3qLKizpWKfgKRFO8hiSwufQ8VhLeGL7EerW8cRkPuevzzbUv8p1L74smoWNMNAFFTgWT4ZybEMFKusnMAXt3rmoUSrNzUJRSR1hJ6/o9W786Op5QQ2Zu4HhN/sKG3DrxmCQ0rRn5VXI/wOPJTJeOXSSRDC9K4foUboTzJeeyy6yWbyJ1R7LmIRhWi5usFctEji9RAUROBU04p4gwcuvjhxHL2xffhzFm0/sismu0SqNQfD1B+nFrR5mI+T+dCIKqn2guU5JIRscu0LFzWElp2S4PtB/nTvkmlc9pchoagrnM1b2KI8eVaAKKnBqmdagCQBivRufT88xnF3ac/CfzBc5nl3lz9DJ9PM4VeByGBCMWp7tH+DThlMdHUQiWpA77DK8D2+VzEzsYwaCqeCqspFhS1ITchTdHLzGoVnEaKolOdj67WybH3UoKQgi93aHlZuT+ERVA5FSxNZxzp9yAoVulZbvbHJQ75hFIypW5d/Ha4EUG1R1U3aYJcjfChLu9JPTREHYvCZaKjf4De+FxoDKuiQTBse6o6Jkl5pJzdU9gQTUUt2vbebw6BMt8dn7Pe0yOu/Ml/eoOAL1kCWuS2GvgAIkmoMipZWtuQPNQ1W0VK6eeC6xXy3z3ub+DNSmmDpHci8mqnscHxUiKwdyFVIqnpNSc0ucUfggCveQcmemQ2U7d+N6HCqS+IHfreHUsZhfr0hrTmRz3lumSuwFWEqwk5L5fX3v7/1fk/hAVQOTUMi03YKccgJ3OVTyqjkG5wmuDa/SSxXFF0Fk4Xg5gg6UVJmrcNqf4Xnh1OC3oJou8Ze5vhYb26inq0hpIiBRSwKlDRLjcfdueppvJcW8irhoF4NVRutFMORuReyOagCKnlmm5ARvH84nnm8+dbHbitGQ06FP4UZ0kddzs+7OglAzGyutuahcJIQdiPr1Iy3boJovcGr3M7fJVnK8odFg31EkQMSxmD5CaLm+MXprqa2mYHPedcjXcxKo/Vhi9/8QdQOTUMi03YON4a+L5xrlbm51YElR8fcRhsNvq4G9wXL9SOvPOZTNCIgnWhFaSlZYojqFbC5nEJjTOSUxW+5WVTrJIZtszrdpt3Tt5WK3WSXtbWmhOKI9YYfT+E3cAkVPLZG7A5OSxUw5AJwmOy9DTV3C+2tQlTNRiTTIuBzHRSYXwzCAYDIZq3KHrdKAa1MegWsOahLVimY6dJ/cDCjcM/QwAIwYrKZXPyWwX2H3VPqrWeWP4TYbVGqCIGkoNGcciBiOW1LZjhdEDJCqAyKllW6nnmq0VK5soFFXP0K2FgmeUYQdgUjrJAiO3jqVL398mlH/ebAJSfGiJeMom//CbVZR1XwTvNDhqSeilS2TSZr1aITUZRkJHtEmzzbRV+9j5C5xvPTw2txmfMPJ9MtOmZ8+Ru0GsMHqARAUQOdVMyw1oJpPJKJReeo5uusigvMNq8SYisJQ9SKVFrUC0To1qSiabOswz7ATutY7/4TN75zHB4PEkJHWv4RGlU0asUvkCayyJpDSlMCCYbfZatTfO36ZC67nWZUo3wqmj8jkXu2+hZXqxwugBExVA5NSzNTdgkq0TkWDoJAsMqjtUvmDgVukmCxix5NUwdPDCkhiLqg/JUdpiqHeCLZwMBSp2d0AfLXcXBBpGxVL6EZnpYE3KoKpQ0U2WsEKHgOBxJJLsumrf6nQXzNhsNHIhoex8++F7+u0isxMVQOTMsFNXsGnRP4ritGK1uMmgWqVtexQyBAWRcK0m81XxUEEiGSKhjELo8XtaTEFKZjsoG03gU9NGVVHxOC3p2EU6Zp4LrUfppUt7rtrvxkEfOTiiAoicCbZm+aqGqJieXaT0OS3TBWEc/ZOZDgZTO4chMQlPzH0f19Y+z6QpqKwKqrrxi9MSS1qbQk7L5B8YVetkSWf82mCYb11AEIZunQvtR3m4t3fcf8PdOOgjB8e+FICI/Azwr4HvBt6rqs9OOe8lYA1wQKWqV/dz30jkbtia5dus8istWC9DfaDc9enZxbr7VRZCHCWhmywhEgrJVeQYMZR+FGrq6GZHsMed2mxVR4HzKalt1WMTehiLBB/BYnbxruz0szroIwfLfncAzwE/DfzmDOf+qKq+uc/7RSJ3zaSdX9WzWtwMUTuSUPmCTrJA4QbcKd8ABCRMUAvZxXEPYFVlOX8NIxYRg1XBUzUNFo/wtzscFCj9ECNCIq3x2Oxnxb6Xgz5y8OxLAajqVwBEjmfbu0gENjscCz+i0oLS56iGej1aKqltk0kHRZhLz5GZjQbwAJUWCDCXLrGch3o4qictG/jeMVjaZo6W7ZKaNpUWVK7Y94p9Nwd95OA5LB+AAp8UEQV+U1U/Nu1EEfkw8GGAK1euHJJ4kdPMpMOxcMOxyUHqeH5HhVWP4ujahfEqv6H0OYJg6/o3iclwrsSP8wHOAsHhe6n9OHPZ+bhiPyXsqQBE5FPAQzsc+qiq/tGM9/lhVb0hIpeAPxWRr6rqX+x0Yq0cPgZw9erVs/LtihwgjcOxcEMG1UodytnU8wnu3MqPELGca11m6NcZVmtUWuC9I7UtHuo8yevDawyq1XpHca+lFU4m3WQJpyXLxQ0e7L01TvqnhD0VgKq+f783UdUb9c83RORp4L3AjgogErnfNA7Hb6x+to4CCjV9IBQ6c1qFloZiaCcLnE8f5aXVL6Dqwm4AeH10jdXiJiCn1tE7jZbp4bUiqVtr9suVaLY5JRx45SoR6YnIfPMc+HGC8zgSOTTayRyX2o/XE7rFEH42jVE8FZnpkpqMV/pfJTEtFrJLdJJ5hm6dteIWlZZ1stPps/yntDEYGD8ahMoXFD5k6RZuwPLoBmvFrTOnCE8j+w0D/SngPwEXgT8WkS+o6j8QkYeB31LVp4AHgadrR3EC/HdV/T/7lDsSuWs8DmMsqWZ4rxvvNT/FgrARMTRRFdRKSkmOrb8yofl7+OTJRxBjSOmQmjaC0K9WxmYywZDQYuT6oZ8yhoG7Ezt1nQL2GwX0NPD0Du/fAJ6qn18Dvmc/94lE7gdWUgRLKimFDMaVPpvG7wIMy7Xx+aUbjfMCqIu/ORymro9z+PuAnRLMJiPw7kYeISFFJCExIXktjIen8uGnSIJBKLWg0rL+lJL7Pl1ZHHfqenLhPdEncEI5rsXLI5H7Tst2Q1eviZVtU9u/Y+dJTYtb+XWaidRNNCgxYsdO39BNSyYeh0NTcnozOvGY/Tot6bLUepCHum8FQgKWlYQwxXsEE8JlNa9/bx9GS1J8nUuRSBo7dZ1wYimIyJmhly7RSnqAUFVF8ANIaGgiInTSBUZVH1Wl9Hk9IQYqX2DqNoXBgbzRByBMmMGfMHsbSKFFF4+jvIvCcS3To/L5uPzEvdBU2ewmSwyrVQAy06kjpBi3jaRWkkFai5UUCCWyvTqKOr8iduo6ucQdQOTM0EQDaW2+CX6p8HMhvRhW2CKcbz2CiNSlHTyFD1VAM9Oll5zDkGBqR3LzSCSp9wK77wjGJZNJcFLdZR6BUPrROHx167GdP7H9K176IdZk5D60iOzaUP3Ua4XXRqYmx9nX1/Y4DRFUplaMTQvHWLjt5BJ3AJEzwWQl0AfaV/AjTyotbN11amOiDArhXHYZpyVL2WWW81dwvmTg7qBq6CbzdOw8peaslbdQVbx6/MROoGmQ3lwTDJmEVffIrY9NSqWbffUcPBBbJ//dFI7skKsQonz65TIXsncy13mAb68/R+GHGNKg9MaTfvgNLIY6Zzq8X/9iHqUVC7edaKICiJx6dqoEWrgBWdIe16BvzluvbqPKeHegqpzPHgYDNwffxpiErp0HgWG+Tma6gJJIi9wPcFqCKkIydhQHx7Fwof1YSCJzYYVd+XKcj3A3SO172GjsPm0XsbPD2FOharg+eIELbj3sKtSRmBZGLV4rKspN5q2O7VD5HEdF7vsYSchMOxZuO+FEBRA51WytBLqBsF4tgzQramW9uk0vOU8nmagY6gvWymXm0wtkto0CuR9Qujz0FZA2C9lFnJbk+TqqHhEhM20Qg6qvbeQais1pqCvkfFXnIFsMyYy+g61mmQ0fxGafxDSncHi/OerVU2g+vlYT1+9r239aV0W1JkFESGwLPLRMh7fMfw/n2pfj5H/CiQogcqrZ2vGroZPMAcr51iOkpl2vgsP7k/H/menUDmA7zgq+1HkrK/mrUMJccg6nFevlbUpf4HGICiM3oJssoCIkJiV3faggS7pk0mGo67VRpSQzXSpva7PRdtPRzrkGQXElklJpMfYlNCYfUysWRzEueKF1H4NGBfi6wF1qWpSmNY4CSghKSwERw1L2EEYMTh1OC67Mv5vF7NJ9/p+KHAVRAURONVs7fk0iIqSmzfn2wyyPXhlXtd0c/x/wWtGy3dp+bzjffoSBWw1NZIqbALTtHEO3RrPSHro1WqYbfAR4hIrCBcerGUcNeXrJOXrpIqXLWSleB4R20mW9WMbX/7bJXoeiOi0nInXCe1Lb/kPUUlWbigyM7fha7xRA1dURQQshqkcBbUpgO1pJj1bSDc1vfE4qLebTWAbitBAVQORUsFO7RyN25taDk+dNxv83mImQ0NLnLLUeJDEZw3J1rCwMFisJqkrLdvDqQ8Zw3TvYsNFbwFORSJuWdEhMipGEVpLQ1UWEkLSWmW4wLakbRw8JYafQmHoUDSt3tVgzEbaqBSKCJR3Lwaa9hQIer2CN8GjvHdzKX2Hk1kMBPGlT+ZyOnSd3AyA2azmNRAUQOfFsdfLCxmQ1a+vByfMm4/83ul9tKIjUtMYhpS/e+eva8RuONa0lUfAUOA+pybAkpNLCU1G5ikIHoKDixyv40ue0bQ9ByP2o3qG0cK4aKyUjBq9ay2NwWobJWka1/8HgNVyzY+cYuXVKDT6ISUydVDaoVjifc5HW1gAACnxJREFUPMK59mXOtS9vas7SSeYZVmux9PMpJiqAyIlmmpO39Pm4TMEsrQcnWxQ6reP/dUgi2dTuV+1kjsfm3s231v+GRNJxZAyExjNr+ZtYm7LYushK8TpeK5w6KkIsf0WB9xWr5U2clmS2w7nsMv3yNmvVct2HICgHS0Jm23j1GGNYzC7htML5KuQp+IySEVo7l62kINCyPSpf4igxWFztbPZ4LBYR4ULr4fE4bK3y2byetsOKnGyiAoicaKY5eVPTYuTWx6WLZ2k9ONmisJ+tcCu/Pi6JMK371Xx2nk4yj6pu2mEYsXSzRYwYjCT0kiXeHL6Moxw7bA2GufQ8VtKgEHzJtbXPbziCBVBP1y5S6ig4mMXSTRYxJuGx3rt4pf9VMt9F8bg6rLRyJbn2QQ25Xw3NberQ1oQsGI7Us5A9EHIg9pjId9thxUJwJ5uoACInmt2cvOF4SLSatfVgc958doFL3cf3VBq7NTd/Yv57eWXwAoUf0q9WSG0LdU2ZheC07VcrtG2PSgtCVE+GlbCLcFrh1ZHZDo923olSYSWjZTtjWZp7V77CGIshNLBfSC8BntKnJDbDOwcImekEMxJ+XPlzt0zeWXZYcSdwcokKIHKimdXJey/MqjR2a27+WO8dXFv9HKUf4X1IChOEVLLaXu8ofcgpSKSNTTa+ksGhHBRGJ+ltk6X5bJO1bCXDaTEOZ22ct4YEKxbvHUZCQTfVIkQA2e6umbyz7rAiJ5OoACInmlmdvAfNNGXRTua41HkiNJNxI6oqJ5FsHHLaEEw+O2f0eu+2FVybZpaZTy6Mr52ZNkYsHkfL9BjqGqUvcFIBQmY7e0b1zLrDipxMogKInGh2M8Ecl5DFlu2SmhaWhNE4T2BzPR8B0J3r+hhjNym33cwyy/kr42uLGBayi6wWN/HqsJLQtl0S0+Lh7ts41354z/E5yB1W5OiJCiBy4tnNBHMcaHYpuQ5JpY2jCGWl67IRqQkJZ0Kw+zdhqE4rkJBgNrmT2c0s47TA1+UnUtMiMRnnWpcZuDVUHVfm38V8emHmsTkuO6zIwRAVQORUMKu9/ihodinfXn+OvoDRBAxYFeaz8yTSCo5hVVbLN6k0mFUEy0J2kStz79w0Ye9ulhEutB5lrbq1aUfUqs09dxu1cxJ2WJF7Z789gX8V+IdAAXwD+Oequq09kIh8APgPgCX0Cv53+7lvJHLSaCdzfMfi3+b26FVeHXwNRTGS4rWkVMfl7ttZbF2iX62wViwDMJ9eYD47v22S3css00uXZopguhvZj/MOK3LviOq99zUVkR8H/kxVKxH5FQBV/eUt51jga8DfB64DnwH+qao+v9f1r169qs8+++w9yxeJHEe8Om6PbnBj8HUUHTuFZ42t9+r4xupnt+UelD5HRGJo5hlHRD6rqldnOXdfHcFU9ZOq48IpnwYe3eG09wIvquo1VS2A3wM+uJ/7RiInneXiVVq2x3x6gU4yT9vOjZusN2WZp9GYZUSEkVsfP0QkmmUid8X99AH8PPD7O7z/CPDyxOvrwA/cx/tGIieK+xFbH80ykfvBngpARD4FPLTDoY+q6h/V53wUqIDf2ekSO7w31e4kIh8GPgxw5cqVvcSLRE4c9yu2/jg7viMngz0VgKq+f7fjIvIh4CeAH9OdHQrXgccmXj8K3Njlfh8DPgbBB7CXfJHISSPG1keOC/vyAdTRPb8M/KSqDqac9hngbSLyhIhkwM8CH9/PfSORk8xkbP0kMbY+ctjsSwEAvwbMA38qIl8Qkd8AEJGHReQZgNpJ/EvAJ4CvAH+gql/e530jkRNLdOJGjgv7cgKr6ndMef8G8NTE62eAZ/Zzr0jkNBGduJHjQMwEjkSOiOjEjRw1+zUBRSKRSOSEEhVAJBKJnFGiAohEIpEzSlQAkUgkckaJCiASiUTOKFEBRCKRyBllX+WgDxoRuQl865Bu9wDw5iHd626Jst0bx1W24yoXRNnuheMm11tU9eIsJx5rBXCYiMizs9bQPmyibPfGcZXtuMoFUbZ74bjKNQvRBBSJRCJnlKgAIpFI5IwSFcAGHztqAXYhynZvHFfZjqtcEGW7F46rXHsSfQCRSCRyRok7gEgkEjmjnFkFICK/KiJfFZG/EZGnRWTHLhwi8gEReUFEXhSRjxySbD8jIl8WES8iU6MLROQlEflS3Yvh2WMm21GM23kR+VMR+Xr989yU81w9Zl8QkQNrTrTXGIhIS0R+vz7+VyLy+EHJcg+y/ZyI3JwYp39xSHL9VxF5Q0Sem3JcROQ/1nL/jYh8/2HINaNsPyIidybG7F8dlmz3jKqeyQfw40BSP/8V4Fd2OMcC3wDeCmTAF4F3HIJs3w18J/DnwNVdznsJeOCQx21P2Y5w3P498JH6+Ud2+j+tj60fgix7jgHwL4HfqJ//LPD7h/R/OItsPwf82mH+bdX3/bvA9wPPTTn+FPAnhF7j7wP+6hjJ9iPA/z7sMdvP48zuAFT1kxq6lQF8mtCreCvvBV5U1WuqWgC/B3zwEGT7iqq+cND3uRdmlO1Ixq2+x2/Xz38b+EeHcM9pzDIGk/L+IfBjIiLHRLYjQVX/Alje5ZQPAv9NA58GlkTk8jGR7cRxZhXAFn6esKrYyiPAyxOvr9fvHRcU+KSIfFZEPnzUwkxwVOP2oKq+ClD/vDTlvLaIPCsinxaRg1ISs4zB+Jx6MXIHOIwOMbP+//zj2szyhyLy2CHINQvH/Tv5gyLyRRH5ExF551ELsxenuiOYiHwKeGiHQx9V1T+qz/koUAG/s9MldnjvvoRNzSLbDPywqt4QkUuEvsxfrVcpRy3bkYzbXVzmSj1ubwX+TES+pKrfuB/yTTDLGBzYOO3BLPf9X8DvqmouIr9I2Kn8vQOXbG+Oasxm4XOEMgzrIvIU8D+Btx2xTLtyqhWAqr5/t+Mi8iHgJ4Af09qIt4XrwOTK51HgxmHINuM1btQ/3xCRpwlb+30rgPsg25GMm4i8LiKXVfXV2izwxpRrNON2TUT+HPg+gk38fjLLGDTnXBeRBFjkcEwMe8qmqrcmXv4Xgp/sOHBgf1v7RVVXJ54/IyL/WUQeUNXjVCdoE2fWBCQiHwB+GfhJVR1MOe0zwNtE5AkRyQiOugOLGrkbRKQnIvPNc4JTe8fohCPgqMbt48CH6ucfArbtVkTknIi06ucPAD8MPH8AsswyBpPy/hPgz6YsRA5dti129Z8EvnIIcs3Cx4F/VkcDvQ+405j9jhoReajx4YjIewnz663dP3XEHLUX+qgewIsEW+IX6kcTjfEw8MzEeU8BXyOsED96SLL9FGGlkwOvA5/YKhshguOL9ePLx0m2Ixy3C8D/Bb5e/zxfv38V+K36+Q8BX6rH7UvALxygPNvGAPg3hEUHQBv4H/Xf4l8Dbz2McZpRtn9b/119Efh/wHcdkly/C7wKlPXf2S8Avwj8Yn1cgF+v5f4Su0TJHYFsvzQxZp8GfuiwZLvXR8wEjkQikTPKmTUBRSKRyFknKoBIJBI5o0QFEIlEImeUqAAikUjkjBIVQCQSiZxRogKIRCKRM0pUAJFIJHJGiQogEolEzij/H8C2cfBk4WalAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x18a8d322ac8>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot the points with colors\n",
    "for k, col in zip(unique_labels, colors):\n",
    "    if k == -1:\n",
    "        # Black used for noise.\n",
    "        col = 'k'\n",
    "\n",
    "    class_member_mask = (labels == k)\n",
    "\n",
    "    # Plot the datapoints that are clustered\n",
    "    xy = X[class_member_mask & core_samples_mask]\n",
    "    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=col, marker=u'o', alpha=0.5)\n",
    "\n",
    "    # Plot the outliers\n",
    "    xy = X[class_member_mask & ~core_samples_mask]\n",
    "    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=col, marker=u'o', alpha=0.5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Practice\n",
    "To better underestand differences between partitional and density-based clusteitng, try to cluster the above dataset into 3 clusters using k-Means.  \n",
    "Notice: do not generate data again, use the same dataset as above."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# write your code here\n",
    "\n"
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
    "\n",
    "from sklearn.cluster import KMeans \n",
    "k = 3\n",
    "k_means3 = KMeans(init = \"k-means++\", n_clusters = k, n_init = 12)\n",
    "k_means3.fit(X)\n",
    "fig = plt.figure(figsize=(6, 4))\n",
    "ax = fig.add_subplot(1, 1, 1)\n",
    "for k, col in zip(range(k), colors):\n",
    "    my_members = (k_means3.labels_ == k)\n",
    "    plt.scatter(X[my_members, 0], X[my_members, 1],  c=col, marker=u'o', alpha=0.5)\n",
    "plt.show()\n",
    "\n",
    "\n",
    "-->"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "\n",
    "\n",
    "<h1 id=\"weather_station_clustering\" align=\"center\"> Weather Station Clustering using DBSCAN & scikit-learn </h1>\n",
    "<hr>\n",
    "\n",
    "DBSCAN is specially very good for tasks like class identification on a spatial context. The wonderful attribute of DBSCAN algorithm is that it can find out any arbitrary shape cluster without getting affected by noise. For example, this following example cluster the location of weather stations in Canada.\n",
    "<br>\n",
    "DBSCAN can be used here, for instance, to find the group of stations which show the same weather condition. As you can see, it not only finds different arbitrary shaped clusters, can find the denser part of data-centered samples by ignoring less-dense areas or noises.\n",
    "\n",
    "let's start playing with the data. We will be working according to the following workflow: </font>\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### About the dataset\n",
    "\n",
    "\t\t\n",
    "<h4 align = \"center\">\n",
    "Environment Canada    \n",
    "Monthly Values for July - 2015\t\n",
    "</h4>\n",
    "<html>\n",
    "<head>\n",
    "<style>\n",
    "table {\n",
    "    font-family: arial, sans-serif;\n",
    "    border-collapse: collapse;\n",
    "    width: 100%;\n",
    "}\n",
    "\n",
    "td, th {\n",
    "    border: 1px solid #dddddd;\n",
    "    text-align: left;\n",
    "    padding: 8px;\n",
    "}\n",
    "\n",
    "tr:nth-child(even) {\n",
    "    background-color: #dddddd;\n",
    "}\n",
    "</style>\n",
    "</head>\n",
    "<body>\n",
    "\n",
    "<table>\n",
    "  <tr>\n",
    "    <th>Name in the table</th>\n",
    "    <th>Meaning</th>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td><font color = \"green\"><strong>Stn_Name</font></td>\n",
    "    <td><font color = \"green\"><strong>Station Name</font</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td><font color = \"green\"><strong>Lat</font></td>\n",
    "    <td><font color = \"green\"><strong>Latitude (North+, degrees)</font></td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td><font color = \"green\"><strong>Long</font></td>\n",
    "    <td><font color = \"green\"><strong>Longitude (West - , degrees)</font></td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>Prov</td>\n",
    "    <td>Province</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>Tm</td>\n",
    "    <td>Mean Temperature (°C)</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>DwTm</td>\n",
    "    <td>Days without Valid Mean Temperature</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>D</td>\n",
    "    <td>Mean Temperature difference from Normal (1981-2010) (°C)</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td><font color = \"black\">Tx</font></td>\n",
    "    <td><font color = \"black\">Highest Monthly Maximum Temperature (°C)</font></td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>DwTx</td>\n",
    "    <td>Days without Valid Maximum Temperature</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td><font color = \"black\">Tn</font></td>\n",
    "    <td><font color = \"black\">Lowest Monthly Minimum Temperature (°C)</font></td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>DwTn</td>\n",
    "    <td>Days without Valid Minimum Temperature</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>S</td>\n",
    "    <td>Snowfall (cm)</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>DwS</td>\n",
    "    <td>Days without Valid Snowfall</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>S%N</td>\n",
    "    <td>Percent of Normal (1981-2010) Snowfall</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td><font color = \"green\"><strong>P</font></td>\n",
    "    <td><font color = \"green\"><strong>Total Precipitation (mm)</font></td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>DwP</td>\n",
    "    <td>Days without Valid Precipitation</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>P%N</td>\n",
    "    <td>Percent of Normal (1981-2010) Precipitation</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>S_G</td>\n",
    "    <td>Snow on the ground at the end of the month (cm)</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>Pd</td>\n",
    "    <td>Number of days with Precipitation 1.0 mm or more</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>BS</td>\n",
    "    <td>Bright Sunshine (hours)</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>DwBS</td>\n",
    "    <td>Days without Valid Bright Sunshine</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>BS%</td>\n",
    "    <td>Percent of Normal (1981-2010) Bright Sunshine</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>HDD</td>\n",
    "    <td>Degree Days below 18 °C</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>CDD</td>\n",
    "    <td>Degree Days above 18 °C</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>Stn_No</td>\n",
    "    <td>Climate station identifier (first 3 digits indicate   drainage basin, last 4 characters are for sorting alphabetically).</td>\n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>NA</td>\n",
    "    <td>Not Available</td>\n",
    "  </tr>\n",
    "\n",
    "\n",
    "</table>\n",
    "\n",
    "</body>\n",
    "</html>\n",
    "\n",
    " "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1-Download data\n",
    "<div id=\"download_data\">\n",
    "    To download the data, we will use <b>!wget</b> to download it from IBM Object Storage.<br> \n",
    "    <b>Did you know?</b> When it comes to Machine Learning, you will likely be working with large datasets. As a business, where can you host your data? IBM is offering a unique opportunity for businesses, with 10 Tb of IBM Cloud Object Storage: <a href=\"http://cocl.us/ML0101EN-IBM-Offer-CC\">Sign up now for free</a>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!wget -O weather-stations20140101-20141231.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/weather-stations20140101-20141231.csv"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2- Load the dataset\n",
    "<div id=\"load_dataset\">\n",
    "We will import the .csv then we creates the columns for year, month and day.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
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
       "      <th>Stn_Name</th>\n",
       "      <th>Lat</th>\n",
       "      <th>Long</th>\n",
       "      <th>Prov</th>\n",
       "      <th>Tm</th>\n",
       "      <th>DwTm</th>\n",
       "      <th>D</th>\n",
       "      <th>Tx</th>\n",
       "      <th>DwTx</th>\n",
       "      <th>Tn</th>\n",
       "      <th>...</th>\n",
       "      <th>DwP</th>\n",
       "      <th>P%N</th>\n",
       "      <th>S_G</th>\n",
       "      <th>Pd</th>\n",
       "      <th>BS</th>\n",
       "      <th>DwBS</th>\n",
       "      <th>BS%</th>\n",
       "      <th>HDD</th>\n",
       "      <th>CDD</th>\n",
       "      <th>Stn_No</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>CHEMAINUS</td>\n",
       "      <td>48.935</td>\n",
       "      <td>-123.742</td>\n",
       "      <td>BC</td>\n",
       "      <td>8.2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>13.5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>273.3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1011500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>COWICHAN LAKE FORESTRY</td>\n",
       "      <td>48.824</td>\n",
       "      <td>-124.133</td>\n",
       "      <td>BC</td>\n",
       "      <td>7.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-3.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>104.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>307.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1012040</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>LAKE COWICHAN</td>\n",
       "      <td>48.829</td>\n",
       "      <td>-124.052</td>\n",
       "      <td>BC</td>\n",
       "      <td>6.8</td>\n",
       "      <td>13.0</td>\n",
       "      <td>2.8</td>\n",
       "      <td>16.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>-2.5</td>\n",
       "      <td>...</td>\n",
       "      <td>9.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>11.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>168.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1012055</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>DISCOVERY ISLAND</td>\n",
       "      <td>48.425</td>\n",
       "      <td>-123.226</td>\n",
       "      <td>BC</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>12.5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1012475</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>DUNCAN KELVIN CREEK</td>\n",
       "      <td>48.735</td>\n",
       "      <td>-123.728</td>\n",
       "      <td>BC</td>\n",
       "      <td>7.7</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.4</td>\n",
       "      <td>14.5</td>\n",
       "      <td>2.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>...</td>\n",
       "      <td>2.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>11.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>267.7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1012573</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 25 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                 Stn_Name     Lat     Long Prov   Tm  DwTm    D    Tx  DwTx  \\\n",
       "0               CHEMAINUS  48.935 -123.742   BC  8.2   0.0  NaN  13.5   0.0   \n",
       "1  COWICHAN LAKE FORESTRY  48.824 -124.133   BC  7.0   0.0  3.0  15.0   0.0   \n",
       "2           LAKE COWICHAN  48.829 -124.052   BC  6.8  13.0  2.8  16.0   9.0   \n",
       "3        DISCOVERY ISLAND  48.425 -123.226   BC  NaN   NaN  NaN  12.5   0.0   \n",
       "4     DUNCAN KELVIN CREEK  48.735 -123.728   BC  7.7   2.0  3.4  14.5   2.0   \n",
       "\n",
       "    Tn   ...     DwP    P%N  S_G    Pd  BS  DwBS  BS%    HDD  CDD   Stn_No  \n",
       "0  1.0   ...     0.0    NaN  0.0  12.0 NaN   NaN  NaN  273.3  0.0  1011500  \n",
       "1 -3.0   ...     0.0  104.0  0.0  12.0 NaN   NaN  NaN  307.0  0.0  1012040  \n",
       "2 -2.5   ...     9.0    NaN  NaN  11.0 NaN   NaN  NaN  168.1  0.0  1012055  \n",
       "3  NaN   ...     NaN    NaN  NaN   NaN NaN   NaN  NaN    NaN  NaN  1012475  \n",
       "4 -1.0   ...     2.0    NaN  NaN  11.0 NaN   NaN  NaN  267.7  0.0  1012573  \n",
       "\n",
       "[5 rows x 25 columns]"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import csv\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "filename='weather-stations20140101-20141231.csv'\n",
    "\n",
    "#Read csv\n",
    "pdf = pd.read_csv(filename)\n",
    "pdf.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3-Cleaning\n",
    "<div id=\"cleaning\">\n",
    "Lets remove rows that don't have any value in the <b>Tm</b> field.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
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
       "      <th>Stn_Name</th>\n",
       "      <th>Lat</th>\n",
       "      <th>Long</th>\n",
       "      <th>Prov</th>\n",
       "      <th>Tm</th>\n",
       "      <th>DwTm</th>\n",
       "      <th>D</th>\n",
       "      <th>Tx</th>\n",
       "      <th>DwTx</th>\n",
       "      <th>Tn</th>\n",
       "      <th>...</th>\n",
       "      <th>DwP</th>\n",
       "      <th>P%N</th>\n",
       "      <th>S_G</th>\n",
       "      <th>Pd</th>\n",
       "      <th>BS</th>\n",
       "      <th>DwBS</th>\n",
       "      <th>BS%</th>\n",
       "      <th>HDD</th>\n",
       "      <th>CDD</th>\n",
       "      <th>Stn_No</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>CHEMAINUS</td>\n",
       "      <td>48.935</td>\n",
       "      <td>-123.742</td>\n",
       "      <td>BC</td>\n",
       "      <td>8.2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>13.5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>273.3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1011500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>COWICHAN LAKE FORESTRY</td>\n",
       "      <td>48.824</td>\n",
       "      <td>-124.133</td>\n",
       "      <td>BC</td>\n",
       "      <td>7.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-3.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>104.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>307.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1012040</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>LAKE COWICHAN</td>\n",
       "      <td>48.829</td>\n",
       "      <td>-124.052</td>\n",
       "      <td>BC</td>\n",
       "      <td>6.8</td>\n",
       "      <td>13.0</td>\n",
       "      <td>2.8</td>\n",
       "      <td>16.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>-2.5</td>\n",
       "      <td>...</td>\n",
       "      <td>9.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>11.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>168.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1012055</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>DUNCAN KELVIN CREEK</td>\n",
       "      <td>48.735</td>\n",
       "      <td>-123.728</td>\n",
       "      <td>BC</td>\n",
       "      <td>7.7</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.4</td>\n",
       "      <td>14.5</td>\n",
       "      <td>2.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>...</td>\n",
       "      <td>2.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>11.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>267.7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1012573</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ESQUIMALT HARBOUR</td>\n",
       "      <td>48.432</td>\n",
       "      <td>-123.439</td>\n",
       "      <td>BC</td>\n",
       "      <td>8.8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>13.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.9</td>\n",
       "      <td>...</td>\n",
       "      <td>8.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>12.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>258.6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1012710</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 25 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                 Stn_Name     Lat     Long Prov   Tm  DwTm    D    Tx  DwTx  \\\n",
       "0               CHEMAINUS  48.935 -123.742   BC  8.2   0.0  NaN  13.5   0.0   \n",
       "1  COWICHAN LAKE FORESTRY  48.824 -124.133   BC  7.0   0.0  3.0  15.0   0.0   \n",
       "2           LAKE COWICHAN  48.829 -124.052   BC  6.8  13.0  2.8  16.0   9.0   \n",
       "3     DUNCAN KELVIN CREEK  48.735 -123.728   BC  7.7   2.0  3.4  14.5   2.0   \n",
       "4       ESQUIMALT HARBOUR  48.432 -123.439   BC  8.8   0.0  NaN  13.1   0.0   \n",
       "\n",
       "    Tn   ...     DwP    P%N  S_G    Pd  BS  DwBS  BS%    HDD  CDD   Stn_No  \n",
       "0  1.0   ...     0.0    NaN  0.0  12.0 NaN   NaN  NaN  273.3  0.0  1011500  \n",
       "1 -3.0   ...     0.0  104.0  0.0  12.0 NaN   NaN  NaN  307.0  0.0  1012040  \n",
       "2 -2.5   ...     9.0    NaN  NaN  11.0 NaN   NaN  NaN  168.1  0.0  1012055  \n",
       "3 -1.0   ...     2.0    NaN  NaN  11.0 NaN   NaN  NaN  267.7  0.0  1012573  \n",
       "4  1.9   ...     8.0    NaN  NaN  12.0 NaN   NaN  NaN  258.6  0.0  1012710  \n",
       "\n",
       "[5 rows x 25 columns]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pdf = pdf[pd.notnull(pdf[\"Tm\"])]\n",
    "pdf = pdf.reset_index(drop=True)\n",
    "pdf.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4-Visualization\n",
    "<div id=\"visualization\">\n",
    "Visualization of stations on map using basemap package. The matplotlib basemap toolkit is a library for plotting 2D data on maps in Python. Basemap does not do any plotting on it’s own, but provides the facilities to transform coordinates to a map projections. <br>\n",
    "\n",
    "Please notice that the size of each data points represents the average of maximum temperature for each station in a year.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'mpl_toolkits.basemap'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-12-6b98a7110c83>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[1;32mfrom\u001b[0m \u001b[0mmpl_toolkits\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbasemap\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mBasemap\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mmatplotlib\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpyplot\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mpylab\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mrcParams\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mget_ipython\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrun_line_magic\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'matplotlib'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'inline'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mrcParams\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'figure.figsize'\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;36m14\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m10\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'mpl_toolkits.basemap'"
     ]
    }
   ],
   "source": [
    "from mpl_toolkits.basemap import Basemap\n",
    "import matplotlib.pyplot as plt\n",
    "from pylab import rcParams\n",
    "%matplotlib inline\n",
    "rcParams['figure.figsize'] = (14,10)\n",
    "\n",
    "llon=-140\n",
    "ulon=-50\n",
    "llat=40\n",
    "ulat=65\n",
    "\n",
    "pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) &(pdf['Lat'] < ulat)]\n",
    "\n",
    "my_map = Basemap(projection='merc',\n",
    "            resolution = 'l', area_thresh = 1000.0,\n",
    "            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)\n",
    "            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)\n",
    "\n",
    "my_map.drawcoastlines()\n",
    "my_map.drawcountries()\n",
    "# my_map.drawmapboundary()\n",
    "my_map.fillcontinents(color = 'white', alpha = 0.3)\n",
    "my_map.shadedrelief()\n",
    "\n",
    "# To collect data based on stations        \n",
    "\n",
    "xs,ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))\n",
    "pdf['xm']= xs.tolist()\n",
    "pdf['ym'] =ys.tolist()\n",
    "\n",
    "#Visualization1\n",
    "for index,row in pdf.iterrows():\n",
    "#   x,y = my_map(row.Long, row.Lat)\n",
    "   my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)\n",
    "#plt.text(x,y,stn)\n",
    "plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5- Clustering of stations based on their location i.e. Lat & Lon\n",
    "<div id=\"clustering\">\n",
    "    <b>DBSCAN</b> form sklearn library can runs DBSCAN clustering from vector array or distance matrix.<br>\n",
    "    In our case, we pass it the Numpy array Clus_dataSet to find core samples of high density and expands clusters from them. \n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "\"['xm' 'ym'] not in index\"",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-13-6300e51d4d85>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpreprocessing\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mStandardScaler\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0msklearn\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mutils\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcheck_random_state\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m1000\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 5\u001b[1;33m \u001b[0mClus_dataSet\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpdf\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'xm'\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;34m'ym'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      6\u001b[0m \u001b[0mClus_dataSet\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mnan_to_num\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mClus_dataSet\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      7\u001b[0m \u001b[0mClus_dataSet\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mStandardScaler\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit_transform\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mClus_dataSet\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Continuum1\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   2131\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mSeries\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mndarray\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mIndex\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlist\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2132\u001b[0m             \u001b[1;31m# either boolean or fancy integer index\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2133\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_array\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2134\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mDataFrame\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2135\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_frame\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Continuum1\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m_getitem_array\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   2175\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_take\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mconvert\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2176\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2177\u001b[1;33m             \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mloc\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_convert_to_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2178\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_take\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mconvert\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2179\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Continuum1\\lib\\site-packages\\pandas\\core\\indexing.py\u001b[0m in \u001b[0;36m_convert_to_indexer\u001b[1;34m(self, obj, axis, is_setter)\u001b[0m\n\u001b[0;32m   1267\u001b[0m                 \u001b[1;32mif\u001b[0m \u001b[0mmask\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0many\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1268\u001b[0m                     raise KeyError('{mask} not in index'\n\u001b[1;32m-> 1269\u001b[1;33m                                    .format(mask=objarr[mask]))\n\u001b[0m\u001b[0;32m   1270\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1271\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0m_values_from_object\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: \"['xm' 'ym'] not in index\""
     ]
    }
   ],
   "source": [
    "from sklearn.cluster import DBSCAN\n",
    "import sklearn.utils\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "sklearn.utils.check_random_state(1000)\n",
    "Clus_dataSet = pdf[['xm','ym']]\n",
    "Clus_dataSet = np.nan_to_num(Clus_dataSet)\n",
    "Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)\n",
    "\n",
    "# Compute DBSCAN\n",
    "db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)\n",
    "core_samples_mask = np.zeros_like(db.labels_, dtype=bool)\n",
    "core_samples_mask[db.core_sample_indices_] = True\n",
    "labels = db.labels_\n",
    "pdf[\"Clus_Db\"]=labels\n",
    "\n",
    "realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)\n",
    "clusterNum = len(set(labels)) \n",
    "\n",
    "\n",
    "# A sample of clusters\n",
    "pdf[[\"Stn_Name\",\"Tx\",\"Tm\",\"Clus_Db\"]].head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As you can see for outliers, the cluster label is -1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{-1, 0, 1, 2}"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "set(labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6- Visualization of clusters based on location\n",
    "<div id=\"visualize_cluster\">\n",
    "Now, we can visualize the clusters using basemap:\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'mpl_toolkits.basemap'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-15-814b2feba22e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[1;32mfrom\u001b[0m \u001b[0mmpl_toolkits\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbasemap\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mBasemap\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mmatplotlib\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpyplot\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mpylab\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mrcParams\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mget_ipython\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrun_line_magic\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'matplotlib'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'inline'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mrcParams\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'figure.figsize'\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;36m14\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m10\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'mpl_toolkits.basemap'"
     ]
    }
   ],
   "source": [
    "from mpl_toolkits.basemap import Basemap\n",
    "import matplotlib.pyplot as plt\n",
    "from pylab import rcParams\n",
    "%matplotlib inline\n",
    "rcParams['figure.figsize'] = (14,10)\n",
    "\n",
    "my_map = Basemap(projection='merc',\n",
    "            resolution = 'l', area_thresh = 1000.0,\n",
    "            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)\n",
    "            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)\n",
    "\n",
    "my_map.drawcoastlines()\n",
    "my_map.drawcountries()\n",
    "#my_map.drawmapboundary()\n",
    "my_map.fillcontinents(color = 'white', alpha = 0.3)\n",
    "my_map.shadedrelief()\n",
    "\n",
    "# To create a color map\n",
    "colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))\n",
    "\n",
    "\n",
    "\n",
    "#Visualization1\n",
    "for clust_number in set(labels):\n",
    "    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])\n",
    "    clust_set = pdf[pdf.Clus_Db == clust_number]                    \n",
    "    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)\n",
    "    if clust_number != -1:\n",
    "        cenx=np.mean(clust_set.xm) \n",
    "        ceny=np.mean(clust_set.ym) \n",
    "        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)\n",
    "        print (\"Cluster \"+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7- Clustering of stations based on their location, mean, max, and min Temperature\n",
    "<div id=\"clustering_location_mean_max_min_temperature\">\n",
    "In this section we re-run DBSCAN, but this time on a 5-dimensional dataset:\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "from sklearn.cluster import DBSCAN\n",
    "import sklearn.utils\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "sklearn.utils.check_random_state(1000)\n",
    "Clus_dataSet = pdf[['xm','ym','Tx','Tm','Tn']]\n",
    "Clus_dataSet = np.nan_to_num(Clus_dataSet)\n",
    "Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)\n",
    "\n",
    "# Compute DBSCAN\n",
    "db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)\n",
    "core_samples_mask = np.zeros_like(db.labels_, dtype=bool)\n",
    "core_samples_mask[db.core_sample_indices_] = True\n",
    "labels = db.labels_\n",
    "pdf[\"Clus_Db\"]=labels\n",
    "\n",
    "realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)\n",
    "clusterNum = len(set(labels)) \n",
    "\n",
    "\n",
    "# A sample of clusters\n",
    "pdf[[\"Stn_Name\",\"Tx\",\"Tm\",\"Clus_Db\"]].head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8- Visualization of clusters based on location and Temperature\n",
    "<div id=\"visualization_location_temperature\">\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from mpl_toolkits.basemap import Basemap\n",
    "import matplotlib.pyplot as plt\n",
    "from pylab import rcParams\n",
    "%matplotlib inline\n",
    "rcParams['figure.figsize'] = (14,10)\n",
    "\n",
    "my_map = Basemap(projection='merc',\n",
    "            resolution = 'l', area_thresh = 1000.0,\n",
    "            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)\n",
    "            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)\n",
    "\n",
    "my_map.drawcoastlines()\n",
    "my_map.drawcountries()\n",
    "#my_map.drawmapboundary()\n",
    "my_map.fillcontinents(color = 'white', alpha = 0.3)\n",
    "my_map.shadedrelief()\n",
    "\n",
    "# To create a color map\n",
    "colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))\n",
    "\n",
    "\n",
    "\n",
    "#Visualization1\n",
    "for clust_number in set(labels):\n",
    "    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])\n",
    "    clust_set = pdf[pdf.Clus_Db == clust_number]                    \n",
    "    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)\n",
    "    if clust_number != -1:\n",
    "        cenx=np.mean(clust_set.xm) \n",
    "        ceny=np.mean(clust_set.ym) \n",
    "        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)\n",
    "        print (\"Cluster \"+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
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
