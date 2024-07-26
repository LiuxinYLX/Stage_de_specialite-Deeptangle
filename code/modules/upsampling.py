import numpy as np

def scale_predictions(predictions, scale_factor_x, scale_factor_y):
    scaled_predictions = np.copy(predictions)
    # scaled_predictions[:, :, 0] *= scale_factor_x
    # scaled_predictions[:, :, 1] *= scale_factor_y
    for i in range(scaled_predictions.shape[0]):
        for j in range(scaled_predictions.shape[1]):
            scaled_predictions[i, j, :, 0] *= scale_factor_x
            scaled_predictions[i, j, :, 1] *= scale_factor_y
            
    return scaled_predictions
