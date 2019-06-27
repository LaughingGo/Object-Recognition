/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow/cc/client/client_session.h"



// #include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <math.h>
#include <fstream>
#include <time.h>
#include <dirent.h>
#include <ctime>

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using namespace std;

using namespace tensorflow;
using namespace tensorflow::ops;

// void ReadImage(string image,int & width,int & height)
// {
  
//   cv::Mat img= imread(image);
// }

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = string(data);
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* indices, Tensor* scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor>& outputs,
                      const string& labels_file_name) {
  std::vector<string> labels;
  size_t label_count;
  Status read_labels_status =
      ReadLabelsFile(labels_file_name, &labels, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }
  const int how_many_labels = std::min(5, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
  }
  return Status::OK();
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     bool* is_expected) {
  *is_expected = false;
  Tensor indices;
  Tensor scores;
  const int how_many_labels = 1;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  if (indices_flat(0) != expected) {
    LOG(ERROR) << "Expected label #" << expected << " but got #"
               << indices_flat(0);
    *is_expected = false;
  } else {
    *is_expected = true;
  }
  return Status::OK();
}

//////////////////////////////////////////////
// image preprocessing
//////////////////////////////////////////////
int IMAGE_MIN_DIM = 400;
int IMAGE_MAX_DIM = 512;
int IMAGE_MIN_SCALE = 0;
int CLASS_NUM = 1+5;
float INFERENCE_SCALE=1;


vector<float> generate_pyramid_anchors(float img_shape)
{
  float feature_strides[]={4,8,16,32,64};
  float feature_shapes[5];  
  float scales[]={32, 64, 128, 256,512};
  float ratios[]={0.5, 1, 2};
  float anchor_stride = 1;

  int scale_num=5；
  for(int i=0;i<5;++i)
  {
    feature_shapes[i] = ceil(img_shape/feature_strides[i]);
  }

  vector<float> box;
  int count=0;
  // loop each scale(=5)
  for(int s=0 ;s<5;++s)
  {
      // vector<float> boxes = generate_anchors(scales[i], ratios, shape[i], feature_strides[i], anchor_stride);
      float scale = scales[s];
      float shape = feature_shapes[s];
      float feature_stride=feature_strides[s];      
      float height[3];
      float width[3];
      //  Enumerate heights and widths from scales and ratios
      for(int i=0;i<3;++i)
      {
        height[i] = scale/sqrt(ratios[i]);
        width[i] = scale*sqrt(ratios[i]);
      }
      // Enumerate shifts in feature space
      float* shifts_y =  new float[int(shape)];
      float* shifts_x =  new float[int(shape)];
      for(int i=0;i<shape;++i)
      {
        shifts_y[i]=i*feature_stride;
        shifts_x[i]=i*feature_stride;
      }

      for(int i=0;i<shape;++i)
      {
        for(int j=0;j<shape;++j)
        {
          for(int k=0;k<3;++k)//widths
          {
            float data = shifts_y[i]-0.5*height[k];
            box.push_back(data);
            data = shifts_x[j]-0.5*width[k];
            box.push_back(data);
            data = shifts_y[i]+0.5*height[k];
            box.push_back(data);
            data = shifts_x[j]+0.5*width[k];
            box.push_back(data);
            count+=1;
          }
        }
      }
      delete shifts_y;
      delete shifts_x;      
    }

     
    // norm boxes
    vector<float> norm_boxes;
    vector<float>::iterator begin;
    vector<float>::iterator iter;
    vector<float>::iterator end;
    begin=box.begin();
    end=box.end();
    int tag=0;
    for(iter=begin; iter!=end; iter++)
    {
       float data;
      if(tag<2)
      {
          data= *iter/(img_shape-1);
      }
      else
      {
        data= (*(iter)-1)/(img_shape-1);
      }
      tag+=1;
      tag=tag%4;
      norm_boxes.push_back(data);
    }
    return norm_boxes;
}
Status ReadImage(const cv::Mat& mat,int32& width,int32& height,Tensor& input_image, float& resize_w, float& resize_h)
{
  // cv::Mat mat = cv::imread(img,CV_LOAD_IMAGE_UNCHANGED);
  // cv::imwrite("out2.jpg",mat);
  // mat = cv::imread(img,CV_32FC1);
  // cv::imwrite("out3.jpg",mat);
  // cout<<"channels : "<<mat.channels()<<"rows : "<<mat.rows<<"channels : "<<mat.cols<<endl;
  width = int32(mat.cols);
  height = int32(mat.rows);

  float scale = IMAGE_MIN_DIM /(width>height?height:width);
  scale=scale>1.0?scale:1.0;


  // if IMAGE_MAX_DIM and mode == "square":
  // Does it exceed max dim?
  
  float image_max = float(width>height?width:height);
  if (std::round(image_max * scale) > IMAGE_MAX_DIM)
  {
      scale = IMAGE_MAX_DIM / image_max;
  }
            
  // Resize image using bilinear interpolation
  cv::Size dsize = cv::Size(mat.cols*scale,mat.rows*scale);    
  cv::Mat mat_resize(dsize,CV_32FC3);
  if (scale != 1)
  {      
    cv::resize(mat, mat_resize,dsize);   
  }
  else
  {
    mat_resize = mat;
  }
  
   int w = mat_resize.cols;
   int h = mat_resize.rows;
   resize_w = w;
   resize_h=h;
   int top_pad = (IMAGE_MAX_DIM - h) / 2;
   int bottom_pad = IMAGE_MAX_DIM - h - top_pad;
   int left_pad = (IMAGE_MAX_DIM - w) / 2;
   int right_pad = IMAGE_MAX_DIM - w - left_pad;
   cv ::Mat dst;
  // cout<<"top_pad : "<<top_pad<<"bottom_pad : "<<bottom_pad<<"left_pad : "<<left_pad<<"right_pad : "<<right_pad<<endl;
  
   cv::copyMakeBorder(mat_resize, dst, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT);
  //  cv::imwrite("dst.jpg",dst);
  //  cv::imwrite("mat_resize.jpg",mat_resize);
  //  cv::Rect rect(-50, -50, 300, 300);
  //  cv::Mat crop_im1 = ImageCropPadding(src, rect);

  // convert mat to tensor
  const float * source_data = (float*)dst.data;

  //input image_tensor  as parameter
  // Tensor image_tensor(DT_FLOAT,TensorShape({1,dst.rows,dst.cols,3}));
  input_image = Tensor(DT_FLOAT,TensorShape({1,dst.rows,dst.cols,3}));
  auto input_tensor_mapped = input_image.tensor<float,4>();
  int depth = 3;
  int n=1;
  int sz[]={3,1024,1024};
  cv::Mat mat2 = cv::Mat(1024,1024,dst.type());
  cv::imwrite("test_dst.png",dst);
  cout<<"mat2 channels:"<<mat2.channels()<<endl;
  // const float * source_data2= (float*)mat2.data;
  for (int y = 0; y < dst.rows; ++y) 
  {
    uchar* data = dst.ptr(y);
    // uchar* data_out =mat2.ptr(y);
    for (int x = 0; x < dst.cols; ++x) 
    {
        input_tensor_mapped(0, y, x, 0) = data[x*3+0]-123.7;       
        input_tensor_mapped(0, y, x, 1) = data[x*3+1]-116.8;   
        input_tensor_mapped(0, y, x, 2) = data[x*3+2]-103.9;
        // np.array([123.7, 116.8, 103.9])
        //  data_out[x*3+0] = input_tensor_mapped(0, y, x, 0);   
        //  data_out[x*3+1] = input_tensor_mapped(0, y, x, 1);   
        //  data_out[x*3+2] = input_tensor_mapped(0, y, x, 2);   
        // mat2.at<cv::Vec3b>(y,x)=dst.at<cv::Vec3b>(y,x);
    }
  }
 
  return Status::OK();
}

Status load_anchor_meta_window(int32& width,int32& height,std::vector<Tensor>* anchor,std::vector<Tensor>* meta,float w,float h,float* window)
{  
  //  int w = mat_resize.cols;
  //  int h = mat_resize.rows;
   int top_pad = (IMAGE_MAX_DIM - h) / 2;
   int bottom_pad = IMAGE_MAX_DIM - h - top_pad;
   int left_pad = (IMAGE_MAX_DIM - w) / 2;
   int right_pad = IMAGE_MAX_DIM - w - left_pad;

  //window
  window[0] =  float(top_pad);
  window[1] =  float(left_pad);
  window[2] =  float(h + top_pad);
  window[3] =  float(w + left_pad);
  // generate anchor
  float new_size_w=IMAGE_MAX_DIM;
  float new_size_h=IMAGE_MAX_DIM;
  vector<float> anchors;
  anchors = generate_pyramid_anchors(new_size_w);
  
  int anchor_size=anchors.size()/4;
  
  Scope scope = Scope::NewRootScope();
  Tensor initConstT(DT_FLOAT, TensorShape({1,anchor_size,4}));
  std::copy_n(anchors.begin(), anchors.size(), initConstT.flat<float>().data());
  auto c_anchor = Const(scope.WithOpName("const_anchor"), initConstT);
  auto v_anchor = Variable(scope.WithOpName("var_anchor"), {1,anchor_size, 4}, DT_FLOAT);
  auto init_anchor = Assign(scope.WithOpName("init_anchor"), v_anchor,c_anchor);


  ClientSession session2(scope);
  TF_CHECK_OK(session2.Run({init_anchor}, anchor));
  // anchor = &outputs2;

  // generate image meta
 
  std::vector<float> initConstData2 = {0.0, float(height),float(width), 3.0,new_size_w,new_size_h,3.0,
  float(top_pad), float(left_pad), float(h + top_pad), float(w + left_pad),INFERENCE_SCALE,0.0,0.0,0.0,0.0,0.0,0.0};
  Tensor initConstT2(DT_FLOAT, TensorShape({1,18}));
  std::copy_n(initConstData2.begin(), initConstData2.size(), initConstT2.flat<float>().data());
  auto c2 = Const(scope.WithOpName("const_c2"), initConstT2);
  auto v2 = Variable(scope.WithOpName("var2"), {1, 18}, DT_FLOAT);
  auto init_v2 = Assign(scope.WithOpName("init_v2"), v2, c2);

  // std::vector<Tensor> outputs3;
  // ClientSession session(scope);

  TF_CHECK_OK(session2.Run({init_v2}, meta));
  return Status::OK();
}


//  """Converts boxes from pixel coordinates to normalized coordinates.
//     boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
//     shape: [..., (height, width)] in pixels

//     Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
//     coordinates it's inside the box.

//     Returns:
//         [N, (y1, x1, y2, x2)] in normalized coordinates
// 
// vector<float> get_anchors(float shape)
// {    
//     vector<float> anchors;
//     float config_BACKBONE_STRIDES[]={4,8,16,32,64};
//     float backbone_shape[5];
//     for(int i=0;i<5;++i)
//     {
//       backbone_shape[i] = ceil(shape/config_BACKBONE_STRIDES[i]);
//     }
//     float scales[]={32, 64, 128, 256, 512};
//     float ratios[]={0.5, 1, 2};
//     float anchor_stride = 1;
//     vector<float> anchors = generate_pyramid_anchors(scales, ratios, backbone_shape, config_BACKBONE_STRIDES, anchor_stride);
   
// }

// backup
vector<float> generate_anchors(float scales,float* ratios,float shape,float feature_stride,float anchor_stride)
{
  // """
  //   scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
  //   ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
  //   shape: [height, width] spatial shape of the feature map over which
  //           to generate anchors.
  //   feature_stride: Stride of the feature map relative to the image in pixels.
  //   anchor_stride: Stride of anchors on the feature map. For example, if the
  //       value is 2 then generate anchors for every other feature map pixel.
  //   """
  float height[3];
  float width[3];
  //  Enumerate heights and widths from scales and ratios
  for(int i=0;i<3;++i)
  {
    height[i] = scales/sqrt(ratios[i]);
    width[i] = scales*sqrt(ratios[i]);
  }
  // Enumerate shifts in feature space
  float* shifts_y =  new float[int(shape)];
  float* shifts_x =  new float[int(shape)];
  for(int i=0;i<shape;++i)
  {
    shifts_y[i]=i*feature_stride;
    shifts_x[i]=i*feature_stride;
  }

  vector<float> box;
  for(int i=0;i<shape;++i)
  {
    for(int j=0;j<shape;++j)
    {
      for(int k=0;k<3;++k)//widths
      {
        float data = shifts_y[i]-0.5*height[k];
        box.push_back(data);
        data = shifts_x[j]-0.5*width[k];
        box.push_back(data);
        data = shifts_y[i]+0.5*height[k];
        box.push_back(data);
        data = shifts_x[j]+0.5*width[k];
        box.push_back(data);
      }
    }
  }
  delete shifts_y;
  delete shifts_x;
  return box;
}

cv::Mat ImageCropPadding(cv::Mat srcImage, cv::Rect rect)
{
	//cv::Mat srcImage = image.clone();
	int crop_x1 = cv::max(0, rect.x);
	int crop_y1 = cv::max(0, rect.y);
	int crop_x2 = cv::min(srcImage.cols, rect.x + rect.width); //  
	int crop_y2 = cv::min(srcImage.rows, rect.y + rect.height);
 
 
	int left_x = (-rect.x);
	int top_y = (-rect.y);
	int right_x = rect.x + rect.width - srcImage.cols;
	int down_y = rect.y + rect.height - srcImage.rows;
	//cv::Mat roiImage = srcImage(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));    
	cv::Mat roiImage = srcImage(cv::Rect(crop_x1, crop_y1, (crop_x2 - crop_x1), (crop_y2 - crop_y1)));
 
 
	if (top_y > 0 || down_y > 0 || left_x > 0 || right_x > 0)
	{
		left_x = (left_x > 0 ? left_x : 0);
		right_x = (right_x > 0 ? right_x : 0);
		top_y = (top_y > 0 ? top_y : 0);
		down_y = (down_y > 0 ? down_y : 0);
		//cv::Scalar(0,0,255)
		cv::copyMakeBorder(roiImage, roiImage, top_y, down_y, left_x, right_x, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 255));
		//cv::copyMakeBorder(roiImage, roiImage, top_y, down_y, left_x, right_x, cv::BORDER_REPLICATE);//   
		//cv::copyMakeBorder(roiImage, roiImage, top_y, down_y, left_x, right_x, BORDER_REFLECT_101);  //
	}
	//{    
	//  destImage = roiImage;    
	//}    
	return roiImage;
}
Status ReadTensorFromImageFile_preprocess(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors,std::vector<Tensor>* anchor,std::vector<Tensor>* meta) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  
  // Bilinearly resize the image to fit the required dimensions.
  int new_size = input_height>input_width?input_height:input_width;
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {new_size, new_size}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));

  string output_meta = "image_meta";

  std::vector<float> initConstData = {0.0, 0.0,1280.0, 1280.0};

  Scope scope = Scope::NewRootScope();

  Tensor initConstT(DT_FLOAT, TensorShape({1,1,4}));
  std::copy_n(initConstData.begin(), initConstData.size(), initConstT.flat<float>().data());

  auto c = Const(scope.WithOpName("const_c"), initConstT);

  auto v = Variable(scope.WithOpName("var1"), {1,1, 4}, DT_FLOAT);
  auto init_v = Assign(scope.WithOpName("init_v"), v, c);

  std::vector<Tensor> outputs2;
  ClientSession session2(scope);

  TF_CHECK_OK(session2.Run({init_v}, anchor));
  // anchor = &outputs2;

  ///////////////////////////////////////////////

  std::vector<float> initConstData2 = {0.0, 1280.0,720.0, 3.0,1280.0,1280.0,3.0,0.0, 0.0, 1280.0,1280.0, 1.0,0.0,0.0,0.0,0.0,0.0,0.0};
  Tensor initConstT2(DT_FLOAT, TensorShape({1,18}));
  std::copy_n(initConstData2.begin(), initConstData2.size(), initConstT2.flat<float>().data());

  auto c2 = Const(scope.WithOpName("const_c2"), initConstT2);

  auto v2 = Variable(scope.WithOpName("var2"), {1, 18}, DT_FLOAT);
  auto init_v2 = Assign(scope.WithOpName("init_v2"), v2, c2);

  std::vector<Tensor> outputs3;
  // ClientSession session(scope);

  TF_CHECK_OK(session2.Run({init_v2}, meta));
  // meta = &outputs3;
  /////////////////////////////////////////////


  // tensorflow::Tensor input_meta(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,18}));
  // tensorflow::Tensor input_anchor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,4}));


  return Status::OK();
}
///////////////////////////////////////////////////
//parse output
/////////////////////////////////////////////////
// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetBoundingBox(const std::vector<Tensor>& outputs,Tensor* indices, Tensor* scores) 
{
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "mrcnn_mask/Reshape_1";
  
  // TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  //Tensor initConstT2(DT_FLOAT, TensorShape({1,18}));
  //std::copy_n(initConstData2.begin(), initConstData2.size(), initConstT2.flat<float>().data());
  // const float p = outputs[0][0][0][0];
  // const float q = *(outputs.flat<float>().data())
  // std::unique_ptr<tensorflow::Session> session(
  //     tensorflow::NewSession(tensorflow::SessionOptions()));
  // TF_RETURN_IF_ERROR(session->Create(graph));
  // // The TopK node returns two outputs, the scores and their original indices,
  // // so we have to append :0 and :1 to specify them both.
  // std::vector<Tensor> out_tensors;
  // TF_RETURN_IF_ERROR(session->Run({}, {outputs},
  //                                 {}, &out_tensors));
  // *scores = out_tensors[0];
  // *indices = out_tensors[1];
  return Status::OK();
}

void getFiles( string path, vector<string>& files )  
{  
    DIR *dir;
    struct dirent *ptr;
    char base[1000];
  
    if ((dir=opendir(path.c_str())) == NULL)
          {
      perror("Open dir error...");
                  exit(1);
          }
  
    while ((ptr=readdir(dir)) != NULL)
    {
      if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
              continue;
      else if(ptr->d_type == 8)    ///file
        //printf("d_name:%s/%s\n",basePath,ptr->d_name);
        files.push_back(ptr->d_name);
      else if(ptr->d_type == 10)    ///link file
        //printf("d_name:%s/%s\n",basePath,ptr->d_name);
        continue;
      else if(ptr->d_type == 4)    ///dir
      {
        files.push_back(ptr->d_name);
        /*
              memset(base,'\0',sizeof(base));
              strcpy(base,basePath);
              strcat(base,"/");
              strcat(base,ptr->d_nSame);
              readFileList(base);
        */
      }
    }
    closedir(dir);

  
    //排序，按从小到大排序
    sort(files.begin(), files.end());
    // return files;
}

int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.

  std::cout<<"load data ..."<<endl;
  string image = "/home/jianfenghuang/Myproject/Mask_Rcnn/mask_rcnn_C++/label_image/label_image/data/colorimage_139.jpg";
  //  string image = "/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/PLC_data/color_p1/colorimage_139.jpg";
  string graph =
      "/home/jianfenghuang/Myproject/Mask_Rcnn/mask_rcnn_C++/label_image/label_image/data/mask_rcnn_plc_0150.pb";
  string labels =
      "/home/jianfenghuang/Myproject/Mask_Rcnn/mask_rcnn_C++/label_image/label_image/data/imagenet_slim_labels.txt";
  std::cout<<"load data done!"<<endl;

  int32 input_width = 720;
  int32 input_height = 1280;
  float input_mean = 0;
  float input_std = 1;

  // // read image
  // cv::Mat input_mat = ReadImage(image,input_width,input_height);

  string input_layer = "input_image";
  string input_layer2 = "input_image_meta";
  string input_layer3 = "input_anchors";
  // string output_layer = "mrcnn_mask/Reshape_1";
  string output_layer_mask = "mrcnn_mask/Reshape_1";
  string output_layer_bbox = "mrcnn_bbox/Reshape";
  string output_layer_detection = "mrcnn_detection/Reshape_1";
  string output_layer_class = "mrcnn_class/Reshape_1";
  // detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates

 // <class 'list'>: ['', 'mrcnn_class/Reshape_1', 'mrcnn_bbox/Reshape', 'mrcnn_mask/Reshape_1', 'ROI/packed_2', 'rpn_class/concat', 'rpn_bbox/concat']
  bool self_test = false;
  string root_dir = "";
  std::vector<Flag> flag_list = {
      Flag("image", &image, "image to be processed"),
      Flag("graph", &graph, "graph to be executed"),
      Flag("labels", &labels, "name of file containing labels"),
      Flag("input_width", &input_width, "resize image to this width in pixels"),
      Flag("input_height", &input_height,
           "resize image to this height in pixels"),
      Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      Flag("input_std", &input_std, "scale pixel values to this std deviation"),
      Flag("input_layer", &input_layer, "name of input layer"),
      Flag("output_layer", &output_layer_mask, "name of output layer"),
      Flag("output_layer_bbox", &output_layer_bbox, "name of output output_layer_bbox"),
      Flag("output_layer_class", &output_layer_class, "name of output output_layer_bbox"),
      Flag("self_test", &self_test, "run a self test"),
      Flag("root_dir", &root_dir,
           "interpret image and graph file names relative to this directory"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  
  /////////////////////// loop each image
  vector<string> files;  
  
  ////获取该路径下的所有文件  
  string filePath = "/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/PLC_data/color_p1/";
  getFiles(filePath, files );  
  time_t start ,end;
  double cost;  
  // vector<cv::Mat> img_mats;
  
  // for(int loop=0;loop<files.size();++loop)
  // {
  //   string cur_img = filePath+files[loop];
  //   cv::Mat mat = cv::imread(cur_img,CV_LOAD_IMAGE_UNCHANGED);
  //   img_mats.push_back(mat);
  // }

  float original_w,original_h;
  
  //load anchor and meta
  std::vector<Tensor>  input_meta,input_anchor;
  float window[4];
  
  // for(int loop=0;loop<files.size();++loop)
  int loop=-1;
  while(1)
  {
    ++loop;
    string cur_img = filePath+files[loop];
    // cv::Mat mat = cv::imread(cur_img,CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat mat = cv::imread(cur_img,CV_LOAD_IMAGE_UNCHANGED);
    // time(&start);  
    
    // Get the image from disk as a float array of numbers, resized and normalized
    // to the specifications the main graph expects.
    std::vector<Tensor> resized_tensors;
    
    Tensor img_tensor;

    // read image
    cout<<"img: "<<files[loop]<<endl;   
    clock_t start = clock();
    float resize_w,resize_h;
    Status load_status = ReadImage(mat,input_width,input_height,img_tensor,resize_w,resize_h);
    if(loop==0)
    {
        original_w = mat.cols;
        original_h = mat.rows;
        load_anchor_meta_window(mat.cols,mat.rows,&input_anchor,&input_meta,resize_w,resize_h,window);
        for(int i=0;i<4;++i)
        {
            window[i]= window[i]/IMAGE_MAX_DIM;
        }
    }
    
    end   = clock();
    cout<<"read image cost time--------------------:"<<(double)(end - start) / CLOCKS_PER_SEC<<endl;
    // test input
    // cout<<"test input:"<<"width"<<input_width<<"height"<<input_height<<endl;
    // cout<<"input acnchor:"<< input_anchor[0].vec<float>()(0);
    if (!load_status.ok()) {
      LOG(ERROR) << "load image failed: " << load_status;
      return -1;
    }
    cout<<"load image done!"<<endl;
    string image_path = tensorflow::io::JoinPath(root_dir, image);
    
    std::cout<<"preprocess..."<<endl;


    const Tensor& resized_tensor = img_tensor;
    // const Tensor& resized_tensor = resized_tensors[0];
    // cout<<"img_tensor dim:"<<img_tensor.dims()<<endl;
    const Tensor& meta = input_meta[0];
    const Tensor& anchor = input_anchor[0];
    // auto resized_tensor_height = resized_tensor.shape().dim_sizes()[1];
    // auto resized_tensor_width = resized_tensor.shape().dim_sizes()[2];
    
    // Tensor molded_images,image_metas, windows;
    // mold_inputs(resized_tensor,resized_tensor_height,resized_tensor_width rmolded_images, image_metas, windows);


    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    std::cout<<"runing..."<<endl;
    // std::cout<<meta.SummarizeValue(100)<<endl;
    // std::cout<<anchor.DebugString()<<endl;
    // std::cout<<anchor.SummarizeValue(100)<<endl;
    // std::cout<<resized_tensor.DebugString()<<endl;
    // string s = anchor.DebugString();
    
  
    end   = clock();
    // cout<<"cost time--------------------:"<<(double)(end - start) / CLOCKS_PER_SEC<<endl;
    Status run_status = session->Run({{input_layer, resized_tensor},{input_layer2, meta},{input_layer3, anchor}},
                                      {output_layer_detection,output_layer_class}, {}, &outputs);
      
    
    
    // cout<<"cost time--------------------:"<<(double)(end - start) / CLOCKS_PER_SEC<<endl;
    /////////////////////
    //parse output
    //////////////////////
    Tensor indices;
    Tensor scores;

    // auto output_c = outputs[0].scalar<float>();
    
    tensorflow::TTypes<float>::Flat indices_flat = outputs[0].flat<float>();
    tensorflow::TTypes<float>::Flat indices_flat_class = outputs[1].flat<float>();
    // s = outputs[0].DebugString();
    // ofstream write;
    // write.open("detection.txt");
    // write<< outputs[0].SummarizeValue(1000*6);
    // write.close();
    // std::cout<<"detection.txt done!"<<endl;


    
    // cv::Mat original_img = cv::imread(cur_img,CV_LOAD_IMAGE_UNCHANGED);
    float shift,scale;
    
    for (int pos = 0; pos < 100; ++pos) 
    {
      // tensorflow::TTypes<float>::Flat indices_flat = outputs[0][pos].flat<float>();
      int begin = pos*6;
    
      float y1 = indices_flat(begin);
      float x1 = indices_flat(begin+1);
      float y2 = indices_flat(begin+2);
      float x2 = indices_flat(begin+3);
      float cc = indices_flat(begin+4);
      float s = indices_flat(begin+5);

      float C[6] = {indices_flat_class(begin+0),indices_flat_class(begin+1),indices_flat_class(begin+2),
      indices_flat_class(begin+3),indices_flat_class(begin+4),indices_flat_class(begin+5)};
      float tmp=0;
      int c=0;
      for(int k=0;k<6;++k)
      {
        if(tmp<C[k])
        {
          c=k;
          tmp=C[k];
        }
      }
      
      // const float score = scores_flat(pos);
      // LOG(INFO) << "detection box: "<<pos<< "y1,x1,y2,x2: "<< " "<<y1<<" "<<x1<<" "<<y2<<" "<<x2<<" class:" << c<<" sore:" << s ;   

      if(c!=0)
      {
        
        // LOG(INFO) << "target: "<<pos<< "y1,x1,y2,x2: "<<img_y1<<img_x1<<img_y2<<img_x2<<" class:" << c<<" sore:" << s ;   
      	//Rect(int a,int b,int c,int d) : (a,b) is the coordinates of upper left of the rectangle,c,d is the height and width
        const cv::Scalar color1(0,0,255);

        // denorm box        
        
        shift =  window[0];    
        scale  = window[2]-window[0];
        y1 =(y1-shift)/scale;
        shift =  window[1];    
        scale  = window[3]-window[1];
        x1 = (x1-shift)/scale;
        shift =  window[0];    
        scale  = window[2]-window[0];
        y2 = (y2-shift)/scale;
        shift =  window[1];
        scale  = window[3]-window[1]; 
        x2 = (x2-shift)/scale;
        int tag=0;
        
        int img_x1,img_x2,img_y1,img_y2;
        img_x1 = x1*input_width;
        img_x2 = x2*input_width;
        img_y1 = y1*input_height;
        img_y2 = y2*input_height;

        cv::Rect rec(img_x1,img_y1,img_x2-img_x1,img_y2-img_y1);

        cv::rectangle(mat,rec ,color1);
        // cv::imwrite("box.jpg",img_mats[loop]);
        // cout<<"class: "<<C<<endl;
        
      }
      
      // filter by score
      
    }
    end   = clock();
    double fps= 1/((double)(end - start) / CLOCKS_PER_SEC);
    cout<<"fps:------"<<fps<<endl;
    cv::imshow("detection",mat);
    cv::waitKey(1);
    // time(&end);  
    // cost=difftime(end,start);
        
    
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return -1;
    }
    if (self_test) 
    {
      bool expected_matches;
      Status check_status = CheckTopLabel(outputs, 653, &expected_matches);
      if (!check_status.ok()) {
        LOG(ERROR) << "Running check failed: " << check_status;
        return -1;
      }
      if (!expected_matches) {
        LOG(ERROR) << "Self-test failed!";
        return -1;
      }
    }

    // Do something interesting with the results we've generated.
    Status print_status = PrintTopLabels(outputs, labels);
    if (!print_status.ok()) 
    {
      LOG(ERROR) << "Running print failed: " << print_status;
      return -1;
    }
  }

  

  return 0;
}
