#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EnergyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  EnergyParameter  energy_param = this->layer_param_.energy_param();
  
 
  
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
      this->blobs_.resize(1);
    }
    num_ = bottom[0]->shape(1);    //1000x300
    channels_ = bottom[0]->shape(0);
    M_ = 1;
    N_ = num_;        //300
    K_ = channels_;   //1000
    // Intialize the weight
   /* vector<int> weight_shape(2);
    weight_shape[0] = channels_;
    weight_shape[1] = 1;
    this->blobs_[0].reset(new Blob<Dtype>(1,channels_,1,1));  //weighr_filler 1x1000
    
    
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.energy_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());*/
    
   
    // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void EnergyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
 
  vector<int> top_shape(2);
  top_shape[0] = 1;       //300
  top_shape[1] = num_;  
  top[0]->Reshape(top_shape);            //1x300   
  // Set up the bias multiplier
}

template <typename Dtype>
void EnergyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, K_, (Dtype)1.,   //weight:1000x1   botttom:1000x300      
      weight, bottom_data, (Dtype)0., top_data);
      
}

template <typename Dtype>
void EnergyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  //if (this->param_propagate_down_[0]) {
   
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    //printf("    energy:top_diff %f \n", top_diff[2]);
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, M_, N_, (Dtype)1.,
        bottom_data,top_diff, (Dtype)0.,weight_diff);   //[1000X300],[1X300]
    //printf("    energy:weight_diff %f \n", weight_diff[2]);
 
 printf("energy:top_diff %f \n", top_diff[10]);
 const Dtype* weight_data = this->blobs_[0]->cpu_data();       //1000x1

    //const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasTrans, N_, K_, M_, (Dtype)1.,
        weight_data, top_diff, (Dtype)0.,bottom_diff);     //[1000,1],[1,300],
  printf("energy:weight_data %f \n", weight_data[2]);
  printf("energy:bottom_diff %f \n", bottom_diff[2]);
  //}
  
}

#ifdef CPU_ONLY
STUB_GPU(EnergyLayer);
#endif

INSTANTIATE_CLASS(EnergyLayer);
REGISTER_LAYER_CLASS(Energy);

}  // namespace caffe
