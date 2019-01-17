import imageio
import glob

img_dir = "/home/vikas/Documents/artgan/gen_images"
filenames = glob.glob(img_dir + "/*.png")
filenames.sort()

with imageio.get_writer('movie.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
