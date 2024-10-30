import numpy as np
import matplotlib.pyplot as plt
import math


def PCA(X, out_dim):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        out_dim: the desired output dimension
    Output:
        mu: the mean vector of X. Please represent it as a D-by-1 matrix (numpy array).
        W: the projection matrix of PCA. Please represent it as a D-by-out_dim matrix (numpy array).
            The m-th column should correspond to the m-th largest eigenvalue of the covariance matrix.
            Each column of W must have a unit L2 norm.
    Todo:
        1. build mu
        2. build the covariance matrix Sigma: a D-by-D matrix (numpy array).
        3. We have provided code of how to compute W from Sigma
    Useful tool:
        1. np.mean: find the mean vector
        2. np.matmul: for matrix-matrix multiplication
        3. the builtin "reshape" and "transpose()" function of a numpy array
    """

    X = np.copy(X)
    D = X.shape[0] # feature dimension
    N = X.shape[1] # number of data instances

    ### Your job  starts here ###
    """
        use the following:
        np.linalg.eigh (or np.linalg.eig) for eigendecomposition. it returns
        V: eigenvalues, W: eigenvectors
        This function has already L2 normalized each eigenvector.
        NOYE: the output may be complex value: do .real to keep the real part of V and W
        sort the eigenvectors by sorting corresponding eigenvalues
        return mu and W
    
    """
    # 1. build mu
    mu = np.mean(X, axis=1).reshape(D, 1)
    
    # Center the data
    X_center = X - mu
    
    # 2. build the covariance matrix Sigma
    Sigma = (1/N) * np.matmul(X_center, X_center.T)
    
    # 3. eigendecomposition of the covariance matrix
    V, W = np.linalg.eigh(Sigma)
    
    # 4. Sort the eigenvectors and eigenvalues in descending order
    idx = np.argsort(V)[::-1]
    V = V[idx]
    W = W[:,idx]
    
    # 5. Select the top out_dim eigenvectors
    W = W[:, :out_dim]
  

    return mu, W

    ### Your job  ends here ###


### Your job  starts here ###   
"""
    load MNIST
    compute PCA
    produce figures and plots
"""
data = np.loadtxt('mnist_test.csv', delimiter=',')
labels = data[:, 0]
images = data[:, 1:].T

# select the images of digit 3
digit3_idx = np.where(labels == 3)[0]
images_digit3 = images[:, digit3_idx]

# compute PCA
dimensions = [2, 8, 64, 128, 784]
mu_list = []
W_list = []

for dim in dimensions:
    mu, W = PCA(images_digit3, dim)
    mu_list.append(mu)
    W_list.append(W)
    
# select one image and reconstruct it
test_image = images_digit3[:, 0].reshape(-1, 1)
reconstructed_images = []

for mu, W in zip(mu_list, W_list):
    # Project the test image to the PCA subspace
    y = np.matmul(W.T, test_image - mu)
    # Reconstruct the image from the PCA subspace
    x_reconstructed = mu + np.matmul(W, y)
    reconstructed_images.append(x_reconstructed)
    
# plot the original and reconstructed images
plt.figure(figsize=(15, 3))
plt.subplot(1, 6, 1)
plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.title('Original')
plt.axis('off')

for i, img in enumerate(reconstructed_images):
    plt.subplot(1, 6, i+2)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f'Dim {dimensions[i]}')
    plt.axis('off')

plt.show()

# Compute PCA for all digits
# define the dimensions to evaluate
dimensions_range = range(10, 785, 10)

# Digit 3
images_a = images[:, np.where(labels == 3)[0]]
# Digits 3 and 8
images_b = np.where((labels == 3) | (labels == 8))[0]
images_b = images[:, images_b]
# Digits 3, 8, and 9
images_c = np.where((labels == 3) | (labels == 8) | (labels == 9))[0]
images_c = images[:, images_c]

# Compute PCA for all digits
def compute_pca_for_case(images_case):
    mu_list = []
    W_list = []
    for dim in dimensions_range:
        mu, W = PCA(images_case, dim)
        mu_list.append(mu)
        W_list.append(W)
    return mu_list, W_list

mu_list_a, W_list_a = compute_pca_for_case(images_a)
mu_list_b, W_list_b = compute_pca_for_case(images_b)
mu_list_c, W_list_c = compute_pca_for_case(images_c)

# Compute the average reconstruction error
# select 100 images of Digit3 for test set
test_indices = np.where(labels == 3)[0][:100]
test_images = images[:, test_indices]

# calculate
def calculate_reconstruction_errors(mu_list, W_list, test_images):
    errors = []
    for mu, W in zip(mu_list, W_list):
        X_centered = test_images - mu
        Y = np.matmul(W.T, X_centered)
        X_reconstructed = mu + np.matmul(W, Y)
        # Compute the mean squared reconstruction error
        error = np.mean(np.sum((test_images - X_reconstructed) ** 2, axis=0))
        errors.append(error)
    return errors

errors_a = calculate_reconstruction_errors(mu_list_a, W_list_a, test_images)
errors_b = calculate_reconstruction_errors(mu_list_b, W_list_b, test_images)
errors_c = calculate_reconstruction_errors(mu_list_c, W_list_c, test_images)

# plot the error curves
plt.figure(figsize=(10, 6))
plt.plot(dimensions_range, errors_a, label='Digits 3 Only')
plt.plot(dimensions_range, errors_b, label='Digits 3 and 8')
plt.plot(dimensions_range, errors_c, label='Digits 3, 8, and 9')
plt.xlabel('Number of Principal Components')
plt.ylabel('Average Reconstruction Error')
plt.title('Reconstruction Error vs. Number of Principal Components')
plt.legend()
plt.grid(True)
plt.show()

