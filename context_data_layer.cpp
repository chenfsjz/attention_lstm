#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ContextDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const ConcatParameter& concat_param = this->layer_param_.concat_param();
 
  //    << "Either axis or concat_dim should be specified; not both.";
  M_ = bottom[0]->shape(0);    //1
  N_ = bottom[1]->shape(1);    //500
  K_ = bottom[0]->shape(1);    //300
  
}

template <typename Dtype>
void ContextDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
  // Initialize with the first blob.
  vector<int> top_shape(2);
  
  top_shape[0] = bottom[0]->shape(0);
  top_shape[1] = bottom[1]->shape(1);
  top[0]->Reshape(top_shape);   //1x500 the context vector input to the lstm2
  //CHECK_EQ(bottom_count_sum, top[0]->count());
}

template <typename Dtype>
void ContextDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* probability_data = bottom[0]->cpu_data();
  const Dtype* input_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,  
    (Dtype)1., probability_data, input_data,           //[1,300] ,[300,500]=[1,500]
    (Dtype)0., top_data);       
}

template <typename Dtype>
void ContextDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  //printf("context:hello");
  const Dtype* top_data_diff = top[0]->cpu_diff();
  const Dtype* probability_data = bottom[0]->cpu_data();
  const Dtype* input_data = bottom[1]->cpu_data();
  //printf("    context_top_diff:%f  \n",top_data_diff[100]);      
  Dtype* probability_data_diff = bottom[0]->mutable_cpu_diff();
  Dtype* input_diff = bottom[1]->mutable_cpu_diff();
   //compute the context_diff and input_data_diff
   
   printf("    input_diff:%f  \n",input_diff[100]);
   caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,  //
    (Dtype)1.,top_data_diff,input_data,            //[1,500] ,[300,500]=[1,300]
    (Dtype)0., probability_data_diff);    
   printf("    probability_data_diff:%f  \n",probability_data_diff[100]);
    
    //printf("%f  ",probability_data_diff[2]);                
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,  //
    (Dtype)1., probability_data,top_data_diff,            //[1,300],[1,500] =[300,500]
    (Dtype)0., input_diff); 
  
  for (int i=0; i<bottom[1]->count();++i)
  {
  	input_diff[i] = input_diff[i]+0.0001; 
  	//printf("    context bottom_diff: %f  \n",input_diff[i]);
  }
  printf("    context bottom_diff: %f  \n",input_diff[2]);     
}
#ifdef CPU_ONLY
STUB_GPU(ContextDataLayer);
#endif

INSTANTIATE_CLASS(ContextDataLayer);
REGISTER_LAYER_CLASS(ContextData);

}  // namespace caffe
