from absl import app, flags
from flax import linen as nn # type: ignore
import deeptangle as dt # type: ignore
import matplotlib.pyplot as plt
import os
import numpy as np
float = np.float64
int = np.int_

import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(module_path)

from filtering import filter_predictions
from upsampling import scale_predictions
from plotting import plot_predictions
from frame_extraction import get_output_folder_video, get_output_folder_dossier, extract_frames_from_video, apply_pooling
from image_processing import clip_processing


# Paramètres à régler
flags.DEFINE_string("input", default=None, required=True, help="Path to the video.")
flags.DEFINE_bool("pooling", default=None, required=True, help="Choose to apply pooling or not")
flags.DEFINE_string("output", default="images/resultat/out.png", help="File where the output is saved.")
flags.DEFINE_float("score_threshold", default=0.1, help="Score threshold to prune bad predictions.")
## Nouveaux parametres
flags.DEFINE_float("hausdorff_threshold", default=50.0, help="Hausdorff distance threshold.")
flags.DEFINE_float("distance_threshold", default=10.0, help="Distance threshold for close points.")
flags.DEFINE_integer("nb_threshold", default=5, help="Number threshold for close points.")
FLAGS = flags.FLAGS


# Paramètre fixé 
flags.DEFINE_string("model", default="ckpt", help="Path to the weights")


# Nous modifions rarement ces paramètres
flags.DEFINE_float("correction_factor", default=1, help="Value of the correction_factor.")
flags.DEFINE_float("overlap_threshold", default=0.4, help="Overlap score threshold to suppress predictions.")
flags.DEFINE_integer("frame", default=5, help="Target frame to detect")



def main(args):
    del args

    with dt.time_activity("Loading Model"):
        forward_fn, state = dt.load_model(FLAGS.model)

    output_folder = get_output_folder_video(FLAGS.input)
    output_folder_p4 = get_output_folder_dossier(output_folder,"pool_4")
    output_folder_p8 = get_output_folder_dossier(output_folder,"pool_8")

    if FLAGS.pooling == True :
        with dt.time_activity("Extracting and Applying Pooling to frames"):
            extract_frames_from_video(FLAGS.input, output_folder)
            apply_pooling(output_folder,4)
            apply_pooling(output_folder,8)



    with dt.time_activity("Obtaining one-hand predictions"):
        clip1, predictions1 = clip_processing(
            output_folder_p4,
            forward_fn,
            state,
            FLAGS.score_threshold,
            FLAGS.overlap_threshold,
            FLAGS.correction_factor
        )
       
        clip2, predictions2 = clip_processing(
            output_folder_p8,
            forward_fn,
            state,
            FLAGS.score_threshold,
            FLAGS.overlap_threshold,
            FLAGS.correction_factor
        )
        
    with dt.time_activity("Scaling predictions"):
        scale_factor_x = clip1.shape[3] / clip2.shape[3]  # width scale factor
        scale_factor_y = clip1.shape[2] / clip2.shape[2]  # height scale factor
        scaled_predictions2 = scale_predictions(predictions2.w, scale_factor_x, scale_factor_y)


    with dt.time_activity("User-friendly Filtering predictions"):
        filtered_predictions = filter_predictions(
            scaled_predictions2[:,1], 
            predictions1.w[:,1], 
            FLAGS.hausdorff_threshold, 
            FLAGS.distance_threshold, 
            FLAGS.nb_threshold, 
            clip1)
            
        print(f"Filtered predictions: {len(filtered_predictions)}")
        
    with dt.time_activity("Plotting the results"):
        plot_predictions(clip1, filtered_predictions, "Nombre de vers")
        plt.savefig(FLAGS.output, dpi=300)
        plt.show()
        
        
        
if __name__ == "__main__":
    app.run(main)
