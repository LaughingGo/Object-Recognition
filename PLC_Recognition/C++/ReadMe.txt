1. Run transform_model.py we can transform the h5 weight file to protobuff weight file.

2. Run following command for compiling:
g++ -g -I /usr/local/include/ -I /usr/local/protobuf/include -I /usr/include/third_party/eigen3 -I /usr/include/third_party/eigen3/Eigen -L /usr/local/lib -L /usr/local/protobuf/lib -std=c++11 inference_cc.cc  -ltensorflow_cc  -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

3. Run inference_sample.out we can get the results of inferencing, which is just a sample of the output of compliling.
