import argparse
import numpy as np
import utils

def main(args):
    ## ARGS
    # f_baseline = './dicoms/baseline.txt'
    # f_production = './dicoms/production.txt'
    # num_groups = 5
    # reduce_method = "pca"
    # heatmap = True
    f_baseline = args.f_baseline
    f_production = args.f_production
    num_groups = args.num_groups
    model = args.model
    reduce_method = args.reduce_method
    heatmap = args.heatmap

    features_model, auto_transforms = utils.get_model(model)
    
    #exctract features
    features_baseline = utils.extract_dicom_features(features_model, auto_transforms, f_baseline)
    features_production = utils.extract_dicom_features(features_model, auto_transforms, f_production)

    #plot heatmaps
    if heatmap:
        utils.plot_heatmap(features_baseline, features_production)
    
    #calculate centroids and distances
    centroid_baseline = np.mean(features_baseline, axis=0)

    production_centroids = utils.get_centroids(features_production, num_groups)
    
    cosine_distances, euclidean_distances = utils.get_distances(centroid_baseline, production_centroids, num_groups)
    
    utils.plot_distances(euclidean_distances, cosine_distances)

    #reduce dimensions and do scatter plots
    all_features = np.concatenate((features_baseline, features_production), axis=0)

    baseline_labels = utils.get_labels(features_baseline, 1)
    production_labels = utils.get_labels(features_production, num_groups) + 1
    
    all_labels = np.concatenate((baseline_labels, production_labels), axis=0)

    all_features_2d = utils.reduce_dimensions(all_features, all_labels, reduce_method)

    utils.plot_points(all_features_2d, all_labels)
    utils.plot_points(all_features_2d, all_labels, binary_plot=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    
    parser.add_argument("--f_baseline", type=str, required=True, help="Path to baseline file")
    parser.add_argument("--f_production", type=str, required=True, help="Path to production file")
    parser.add_argument("--num_groups", type=int, default=5, help="Number of groups (default: 5)")
    parser.add_argument("--model", type=str, default="b0", help="Pretrained model to extract features (default: b0)")
    parser.add_argument("--reduce_method", type=str, default="pca", help="Reduction method (default: pca)")
    parser.add_argument("--heatmap", type=bool, default=True, help="Enable heatmap (default: True)")

    args = parser.parse_args()
    main(args)