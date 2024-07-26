import matplotlib.pyplot as plt

    
def plot_predictions(clip, predictions, num_objects_text):
    plt.style.use("fast")
    plt.figure(figsize=(6, 4))
    plt.xlim(0, clip.shape[3])
    plt.ylim(0, clip.shape[2])
    plt.imshow(clip[0, 5], cmap="binary")

    for i, x in enumerate(predictions):
        plt.plot(x[5:-5, 0], x[5:-5, 1], "-")
        plt.annotate(f'{i}', xy=(x[-6, 0], x[-6, 1]), xytext=(5, 5), textcoords='offset points', fontsize=9, color='red')

    num_objects = len(predictions)
    plt.figtext(0.5, 0.98, f'{num_objects_text}: {num_objects}', horizontalalignment='center', verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    
        
