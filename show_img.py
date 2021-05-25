import matplotlib.pyplot as plt
import matplotlib.image as mpimg


testSetImg0 = mpimg.imread('test_set_0.png')
imgplot0 = plt.imshow(testSetImg0)
testSetImg1 = mpimg.imread('test_set_1.png')
imgplot1 = plt.imshow(testSetImg1)
testSetImg2 = mpimg.imread('test_set_2.png')
imgplot2 = plt.imshow(testSetImg2)
testSetImg3 = mpimg.imread('test_set_3.png')
imgplot3 = plt.imshow(testSetImg3)
testSetImg4 = mpimg.imread('test_set_4.png')
imgplot4 = plt.imshow(testSetImg4)

plt.show()