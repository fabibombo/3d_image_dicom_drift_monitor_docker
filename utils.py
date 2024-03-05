import warnings
from pathlib import Path
from typing import Any, Sequence

import highdicom as hd
import monai
import numpy as np
import pydicom
import SimpleITK as sitk
from highdicom.sr.coding import CodedConcept
from pydicom.sr.coding import Code
from typeguard import typechecked

import process_segs.geometry as geometry

from scipy.spatial import distance
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap

import torch
import torchvision

def get_model(model: str):
    device = "cpu"
    
    if model == "b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        features_model = torchvision.models.efficientnet_b0(weights=weights).to(device)

        #remove classifier
        features_model.classifier = torch.nn.Sequential().to(device)
        #features_model.heads = torch.nn.Sequential().to(device)

    return features_model, weights.transforms()

def _get_pixel_array(image: pydicom.Dataset) -> np.ndarray:
    """Get pixel array from dataset with VOI LUT applied.

    Parameters
    ----------
    image: pydicom.Dataset
        A pydicom dataset representing as image.

    Returns
    -------
    Numpy array representing the image pixels with VOI transformation applied.

    """
    slope = getattr(image, "RescaleSlope", 1)
    intercept = getattr(image, "RescaleIntercept", 0)
    return image.pixel_array * slope + intercept

def read_dicom(impath: str):
    
    impath = Path(impath)
    
    if impath.is_dir():
        images = [pydicom.dcmread(f) for f in impath.iterdir()]
        images = geometry.sort_slices(images)
    else:
        images = [pydicom.dcmread(impath)]
    
    # Convert image to 3D array
    im_pixels = np.dstack([_get_pixel_array(ds) for ds in images])
    im_pixels = im_pixels.astype(np.float32)
    
    # Swap rows and columns to match convention
    # rows, columns -> columns, row
    im_pixels = np.transpose(im_pixels, [1, 0, 2])
    
    if len(images) > 2:
        out_of_plane_spacing = geometry.get_slice_spacing(images)
    else:
        out_of_plane_spacing = 1.0
        
    affine = geometry.create_affine_transformation_matrix(
        image_position=images[0].ImagePositionPatient,
        image_orientation=images[0].ImageOrientationPatient,
        pixel_spacing=images[0].PixelSpacing,
        out_of_plane_spacing=out_of_plane_spacing,
    )
    
    spacing = np.array([*images[0].PixelSpacing, out_of_plane_spacing])
    im_meta = {
        "affine": affine,
        "original_affine": affine,
        "spacing": spacing,
        "filename_or_obj": str(impath),
        "original_channel_dim": "no_channel",
        "lastImagePositionPatient": np.array(images[-1].ImagePositionPatient),
        "spatial_shape": im_pixels.shape,
        "space": monai.utils.SpaceKeys.RAS,
    }

    return im_pixels, im_meta
    
def prepare_frame(x, transform, input_shape=(320, 320)):
    # it sets the frame ready for the model prediction
    x = torch.from_numpy(x)
    x = x.unsqueeze(dim=0)
    x = torch.concat((x, x, x), dim=0)
    x = x.unsqueeze(dim=0)
    x = transform(x)
    return x

def get_features(img, model, transform):
    features_array = []
    for idx_frame in range(img.shape[2]):
        frame = prepare_frame(img[:,:,idx_frame], transform)
        features_array.append(model(frame).detach().numpy())
    features_array = np.array(features_array)
    features = np.mean(features_array, axis=0)
    return features

def extract_dicom_features(model, transforms, file):
    features = []

    print("Reading file", file)
    with open(file, 'r') as file:
        for line in file:
            
            dicom_path = line.strip()
            print("    Extracting features:", dicom_path)
    
            im_pixels, im_meta = read_dicom(dicom_path)
    
            #extract features
            image_features = get_features(im_pixels, model, transforms).squeeze()
            features.append(image_features)
    
    return np.array(features)

def get_centroids(features, num_groups = 10):
    #divide in subgroups and calculate centroids
    N = features.shape[0]
    subgroup_length = N // num_groups
    centroids = []
    
    for i in range(num_groups):
        start_idx = i * subgroup_length
        end_idx = (i + 1) * subgroup_length if i < num_groups-1 else N
        subgroup = features[start_idx:end_idx]
        centroids.append(np.mean(subgroup, axis=0))
    return np.array(centroids)

def get_labels(features, num_groups = 10):
    N = features.shape[0]
    subgroup_length = N // num_groups

    labels = np.array([])

    for idx in range(num_groups):
        start_idx = idx * subgroup_length
        end_idx = (idx + 1) * subgroup_length if idx < num_groups - 1 else N

        subgroup = features[start_idx:end_idx]
        labels = np.concatenate((labels, np.full(subgroup.shape[0], idx)))

    return labels

def get_distances(base_centroid, centroids, num_groups):
    cosine_distances = []
    euclidean_distances = []
    
    agg_centroid = 0
    
    for idx in range(num_groups):
        
        agg_centroid += centroids[idx,:]
        centroid_to_compare = agg_centroid/(idx+1)
        
        cosine_distances.append(distance.cosine(base_centroid, centroid_to_compare))
        euclidean_distances.append(distance.euclidean(base_centroid, centroid_to_compare))
    
    return cosine_distances, euclidean_distances

def plot_distances(euclidean_distances, cosine_distances):
    # Data points
    x_values = range(1, len(cosine_distances) + 1)
    
    # Plotting
    fig, ax2 = plt.subplots()
    
    # Plotting on the second y-axis
    ax2.plot(x_values, euclidean_distances, marker='s', color='crimson', label='Euclidean Distances')
    ax2.set_ylabel('Euclidean Distance', color='crimson')
    ax2.tick_params('y', colors='crimson')
    
    # Set x-axis ticks to percentages
    percentage_ticks = np.linspace(100/len(x_values), 100, len(x_values))
    ax2.set_xticks(x_values)
    ax2.xaxis.set_major_locator(ticker.FixedLocator(x_values))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(percentage_ticks[x-1])}%'))
    ax2.set_xlabel('η')

    # Create a second y-axis
    ax1 = ax2.twinx()
    
    # Plotting on the first y-axis
    ax1.plot(x_values, cosine_distances, marker='o', color='green', label='Cosine Distances')
    ax1.set_xlabel('η')
    ax1.set_ylabel('Cosine Distance', color='green')
    ax1.tick_params('y', colors='green')
    
    # Adding title and legend
    plt.title('Cosine and Euclidean Distances')
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(0.86, 0.9))
    
    # Save and show the plot
    plt.savefig('distance_plot.png')
    #plt.show()

#reduce dimensions function
def reduce_dimensions(features, labels, method):
    try:
        # apply PCA
        if method == "pca":
            pca = PCA(n_components=2)
            transformed_features = pca.fit_transform(features)
    
        # TSNE
        elif method == "tsne":
            tsne = TSNE(n_components=2)
            transformed_features = tsne.fit_transform(features)
    
        # LDA
        elif method == "lda":
            lda = LinearDiscriminantAnalysis(n_components=2)
            transformed_features = lda.fit_transform(features, labels)
            
        # UMAP
        elif method == "umap":
            mapper = umap.UMAP().fit(all_features)
            transformed_features = np.transpose(mapper.transform(all_features))

        else:
            raise ValueError(f"Unknown reduction method: {method}")
             
    except Exception as e:
        if "ValueError: perplexity must be less than n_samples" in str(e):
            print("Error: Too few samples for TSNE.")
        else:
            print(f"Error: {e}")
        return None
    
    return transformed_features

def plot_points(features, labels, binary_plot=False):
    if binary_plot:
        labels = np.where(labels > 0, 1, labels)
    
    # Set the size of the plot
    plt.figure(figsize=(8, 6))

    # Create a scatter plot
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', marker='o', label='Data Points')

    # Add grid
    plt.grid(True)

    # Add legend outside the plot to the right
    legend = plt.legend(*scatter.legend_elements(), title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().add_artist(legend)

    # Set labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Features Scatter Plot')

    # Show the plot
    #plt.show()
    plt.savefig('scatter_plot.png', bbox_inches='tight')

def plot_heatmap(features_baseline, features_production):
    # Plot Euclidean differences heatmap
    euclidean_differences = cdist(features_baseline, features_production, metric='euclidean')
    cosine_similarities = cdist(features_baseline, features_production, metric='cosine')
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    heatmap_euclidean = plt.imshow(euclidean_differences, cmap='viridis', interpolation='nearest')
    plt.title('Euclidean Differences')
    plt.colorbar(heatmap_euclidean, shrink=0.8)  # Adjust the shrink parameter
    
    # Plot Cosine similarities heatmap
    plt.subplot(1, 2, 2)
    heatmap_cosine = plt.imshow(cosine_similarities, cmap='viridis', interpolation='nearest')
    plt.title('Cosine Similarities')
    plt.colorbar(heatmap_cosine, shrink=0.8)  # Adjust the shrink parameter
    
    plt.tight_layout()
    #plt.show()
    plt.savefig('cdist_heatmap.png')
