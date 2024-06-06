## Expectation Maximisation for a Gaussian Mixture Model

Implemetation of the expectation maximisation algorithm for Gaussian Mixture Models in C++ based off of [this article's](https://towardsdatascience.com/implementing-expectation-maximisation-algorithm-from-scratch-with-python-9ccb2c8521b3) [1] EM algorithm implentation for Poisson Mixture Models in Python.

After the predicted paramters of the GMM are computed, they are stored as a JSON file to allow for easy export and plotting in Python.

### Setup and run:
Assuming g++ is the installed compiler,

    git clone https://github.com/Linus-J/EM
    cd EM
    g++ EM_GMM.cpp -o EM_GMM
    ./EM_GMM

### Make plots:

Use the Make_Figures.ipynb notebook to generate both true and predicted plots for your GMM.

#### Example output for a GMM with 2 Gaussian Models:
For the distribution $Z_1$, $\mu_1 = 0, \sigma_1 = 1.0, \pi_1 = 0.2.$

For $Z_2$, $\mu_2 = 4, \sigma_2 = 2.0, \pi_2 = 0.8.$


<img src="https://github.com/Linus-J/repo-images/blob/main/EM/Distribution.png" alt="Distribution" width="600"/>
<img src="https://github.com/Linus-J/repo-images/blob/main/EM/True_Distribution.png" alt="True_Distribution" width="600"/>

<img src="https://github.com/Linus-J/repo-images/blob/main/EM/Predicted_Distribution.png" alt="Predicted_Distribution" width="600"/>

### References

[1] Gerry Christian Ongko, Implementing Expectation-Maximisation Algorithm from Scratch with Python, Towards Data Science.
