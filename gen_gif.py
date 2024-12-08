import os
import imageio.v2 as imageio

png_dir = 'movie_materials/marker_tracking'
images = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith(('.jpg','jpeg','bmp','.png')):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))

# Make it pause at the end so that the viewers can ponder
for _ in range(10):
    images.append(imageio.imread(file_path))

imageio.mimsave('./movie/marker_tracking.gif', images, fps=3,loop=True)