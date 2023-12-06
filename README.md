# The problem

An image with a size of `552x344` is given, which depicts several (possible always 7) objects. These objects can be three-dimensional letters, numbers or stereometric shapes. Objects may have different colors and different sizes. There are 2 similar objects in the image, i.e. two identical letters, numbers or shapes. Other objects differ in shape. The task is to use the mouse to indicate the approximate coordinates of the centers of these two similar objects.

![Example](/example.png)

# The dataset for naive implementation

`$DS_ROOT/challenges` directory contains a set of challenges presented as `*.png` images in the form in which they were downloaded from the source.

`$DS_ROOT/solutions` contains solutions (labels) as `*.txt` plain text files. Each file contains two lines. Each line contains two float numbers. These numbers are the normalized coordinates (x, y) of approximate center of one of the similar objects.
