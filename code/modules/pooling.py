from flax import linen as nn # type: ignore
from absl import app, flags
import skimage.io
import jax # type: ignore
import jax.numpy as jnp # type: ignore


class PoolingModel(nn.Module):
    window: tuple
    strides: tuple
    
    def setup(self):
        self.max_pool = nn.max_pool

    def __call__(self, x):
        x = self.max_pool(x, window_shape=self.window, strides=self.strides)
        return x


def max_pooling(input_path, output_path, window, strides):

	# Lecture 
    img = skimage.io.imread(input_path)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.ndim == 3:
        img = skimage.color.rgb2gray(img)
    
    img = (img * 255).astype(jnp.uint8)
    img = jnp.expand_dims(img, axis=(0, -1)) # Ajouter batch et channel
    
	# Appliquer le modèle
    key = jax.random.PRNGKey(0)
    model = PoolingModel(window=window, strides=strides)
    params = model.init(key, img)
    pooled_output = model.apply(params, img)
    pooled_output = jnp.squeeze(pooled_output, axis=(0, -1)) # Enlever batch et channel
    
    # Convert to Numpy
    pooled_output_np = jnp.array(pooled_output)
    pooled_output_np = (pooled_output_np * 255).astype(jnp.uint8)
    
    # Save
    skimage.io.imsave(output_path, pooled_output_np)
    print(f"Output image saved to {output_path}")


