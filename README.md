# Structure from motion

Structure from motion (SfM) is the process of reconstruct- ing the 3D structure of a scene from its projections into a series of images taken from different viewpoints. The code was tested on two provided datasets of images and a new one acquired with a smart- phone camera. The result of this work is a point cloud that represents a 3D structure of a small scene acquired with our smartphone.

<p align="center">
  <img src="media_readme/aloe.jpg" width="45%" />
  <img src="media_readme/cloud2_surf.png" width="45%" />
</p>
<p align="center">
  <img src="media_readme/lego.jpg" width="45%" />
  <img src="media_readme/cloud3_akaze.png" width="45%" />
</p>
<p align="center">
  <img src="media_readme/statua.jpg" width="45%" />
  <img src="media_readme/cloud1_surf.png" width="45%" />
</p>

## To use it
Prerequisites (in debian-based distro):

sudo apt install build-essential cmake libboost-filesystem-dev libopencv-dev libomp-dev
sudo apt install libceres-dev libyaml-cpp-dev libgtest-dev libeigen3-dev

Build and run the executable:

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make

Test the two applications (located inside the bin/ folder)

./matcher <calibration parameters filename> <images folder filename> <output data file> [focal length scale]
./basic_sfm <input data file> <output ply file>

Datasets

In the dataset/ folder there are two simple datasets with a collection of images, the corresponding camera calibration
parameters file, and two preprocessed data files with the results of detection and feature matching for the two datasets
to be used directly with basic_sfm.

Examples

For the provided datasets, set the focal lenght scale to 1.1, e.g.:

./matcher ../datasets/3dp_cam.yml ../datasets/images_1 data1.txt 1.1
./matcher ../datasets/3dp_cam.yml ../datasets/images_2 data2.txt 1.1
./matcher ../datasets/calib_final.yml ../datasets/images_3 data3.txt

./basic_sfm data1.txt cloud1.ply
./basic_sfm data2.txt cloud2.ply
./basic_sfm data3.txt cloud3.ply

To see the cloud

meshlab cloud1.ply
meshlab cloud2.ply
meshlab cloud3.ply
