from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    data = np.load(filename)
    data_mean = np.mean(data , axis = 0)
    data_centered = data - data_mean
    # print(len(data_centered))
    # print(len(data_centered[0]))
    # print(np.average(data_centered))
    return data_centered

def get_covariance(dataset):
    # Your implementation goes here!
    return np.dot((np.transpose(dataset)), dataset)/(len(dataset)-1)

def get_eig(S, m):
    # Your implementation goes here!
    eigen_values, eigen_vectors=  eigh(S, subset_by_index = [S.shape[0]-m , S.shape[0]-1])
    # Rearrange eigenvalues and eigenvectors in descending order
    eigen_values = eigen_values[::-1]
    eigen_vectors = eigen_vectors[:, ::-1]

    # Take the first m eigenvalues and corresponding eigenvectors
    Lambda = np.diag(eigen_values[:m])
    U = eigen_vectors[:, :m]

    return Lambda, U

def get_eig_prop(S, prop):
    eigen_values, eigen_vectors = eigh(S, subset_by_value = [prop*S.trace(), np.inf])
    Lambda = np.diag(np.sort(eigen_values)[::-1])
    U = eigen_vectors[:, :len(eigen_values)]
    return Lambda, U

def project_image(image, U):
    return np.dot(U, np.dot(U.T, image))

def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, ax1, ax2 = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2
    # Reshape the images to be 64x64
    orig = orig.reshape(64, 64)
    proj = proj.reshape(64, 64)

    # Create a figure with one row of two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

    # Title the subplots
    ax1.set_title('Original')
    ax2.set_title('Projection')

    # Display the images on the correct axes
    ax1.imshow(orig, cmap='gray', aspect='equal')
    ax2.imshow(proj, cmap='gray', aspect='equal')

    # Create a colorbar for each image
    cbar1 = fig.colorbar(ax1.imshow(orig, cmap='gray', aspect='equal'), ax=ax1, orientation='vertical')
    cbar2 = fig.colorbar(ax2.imshow(proj, cmap='gray', aspect='equal'), ax=ax2, orientation='vertical')

    # Return the figure and axes for further use
    return fig, ax1, ax2



