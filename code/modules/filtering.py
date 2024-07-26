import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from matplotlib.widgets import RectangleSelector

def nbpt_close(pred1, pred2, distance_threshold, nb_threshold):
    res = False
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    
    close_count = 0
    
    if (min(pred1[:, 0]) > max(pred2[:, 0]) or min(pred2[:, 0]) > max(pred1[:, 0]) or
        min(pred1[:, 1]) > max(pred2[:, 1]) or min(pred2[:, 1]) > max(pred1[:, 1])):
        return False
    else :    
        for i in range(pred1.shape[0]):
            if np.linalg.norm(pred1[i] - pred2[i]) < distance_threshold:
                close_count += 1
            if np.linalg.norm(pred1[-i] - pred2[i]) < distance_threshold:
                close_count += 1
    
        if close_count >= nb_threshold :
            res = True
    
    return res
    
def hausdorff_close(pred1, pred2, threshold):
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    
    if (min(pred1[:, 0]) > max(pred2[:, 0]) or min(pred2[:, 0]) > max(pred1[:, 0]) or
        min(pred1[:, 1]) > max(pred2[:, 1]) or min(pred2[:, 1]) > max(pred1[:, 1])):
        return False
    
    distance_value = distance.directed_hausdorff(pred1, pred2)[0]
    
    if distance_value <= threshold:
        return True
    
    return False

def rectangle_filter(image, predictions, action):
    """
    Filter predictions based on a selected rectangle area.

    Parameters:
    - image: The image to display.
    - predictions: List of predictions to filter.
    - action: 'delete' to omit predictions inside the rectangle, 'select' to keep only predictions inside the rectangle.

    Returns:
    - filtered_predictions: List of filtered predictions.
    """
    _, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size as needed
    ax.imshow(image[0, 5], cmap='binary')
    
    for i, x in enumerate(predictions):
        ax.plot(x[5:-5, 0], x[5:-5, 1], "-")

    rect_coords = []

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        rect_coords.append((x1, y1, x2, y2))
        plt.close()

    rect_selector = RectangleSelector(ax, onselect, interactive=True, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels')

    plt.show()
    
    if rect_coords:
        x1, y1, x2, y2 = rect_coords[0]
        print(f'Selected rectangle coordinates: ({x1}, {y1}), ({x2}, {y2})')
        
        def is_within_rect(x, y):
            return x1 <= x <= x2 and y1 <= y <= y2
            
        filtered_predictions = []
        for pred in predictions:
            within_rect = any(is_within_rect(x, y) for x, y in pred[5:-5])
            if (action == 'delete' and not within_rect) or (action == 'select' and within_rect):
                filtered_predictions.append(pred)
                
        filtered_predictions = np.array(filtered_predictions)
        return filtered_predictions
    
    # Return the original predictions if no rectangle was selected
    return predictions

def filter_predictions(example1, example2, threshold, dist_threshold, nb_threshold, image):
    filtered_predictions = example1.copy()

    for j, pred2 in enumerate(example2):
        add_to_filtered = True
        for i, pred1 in enumerate(example1):
            if nbpt_close(pred1, pred2, dist_threshold, nb_threshold) or hausdorff_close(pred1, pred2, threshold):
                add_to_filtered = False
                break
        
        if add_to_filtered:
            filtered_predictions = np.concatenate((filtered_predictions, [pred2]), axis=0)

    while True:
        user_input = input("Do you want to draw a rectangle to delete or select predictions? (D/S/N):\n D : All the predictions inside the rectangle will be omitted. \n S : Only the predictions inside the rectangle will be kept. \n N : See the results.\n").strip().upper()
        if user_input == 'D':
            filtered_predictions = rectangle_filter(image, filtered_predictions,'delete')
        elif user_input == 'S':
            filtered_predictions = rectangle_filter(image, filtered_predictions,'select')
        elif user_input == 'N':
            break
        else:
            print("Invalid input. Please enter D or S or N.")
    
    return filtered_predictions

