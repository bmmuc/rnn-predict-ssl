import imageio
import numpy as np    
import os


#Create reader object for the gif
gif1 = imageio.get_reader('./gifs/comparacao_15/gif-of-dataset-test-idx-0.gif')
gif2 = imageio.get_reader('./gifs/suposto_over_15/gif-of-dataset-test-idx-0.gif')

#If they don't have the same number of frame take the shorter
number_of_frames = min(gif1.get_length(), gif2.get_length()) 

#Create writer object
new_gif = imageio.get_writer('output-15-steps.gif')

for frame_number in range(number_of_frames):
    img1 = gif1.get_next_data()
    img2 = gif2.get_next_data()
    #here is the magic
    new_image = np.hstack((img1, img2))
    new_gif.append_data(new_image)

gif1.close()
gif2.close()
new_gif.close()
