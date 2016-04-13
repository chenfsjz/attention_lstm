#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype sigmoid_diff(Dtype x) {
  return x * (1. - x);
}

template <typename Dtype>
inline Dtype tanh(Dtype x) {
  Dtype exp2x = exp(2 * x);
  return fabs(x) < Dtype(5) ? ((exp2x - Dtype(1)) / (exp2x + Dtype(1)))
    : (x > 0 ? Dtype(1) : Dtype(-1));
}

template <typename Dtype>
inline Dtype tanh_diff(Dtype x) {
  return (1. - x * x);
}

template <typename Dtype>
void IndtanhLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   
  IndtanhParameter indtanh_param = this->layer_param_.indtanh_param();
  
  //CHECK((lstm_unit_param.has_num_cells()))
  //    << "lstm_unit_param.has_num_cells()";
  CHECK((indtanh_param.has_lstm_hidden_weight_filler()))
      << "indtanh_param.has_lstm_hidden_weight_filler()";
  CHECK((indtanh_param.has_attention_weight_filler()))
      << "indtanh_param.has_attention_weight_filler()";
  

   
  //the lstm input_data 300x500
  num_ = bottom[1]->shape(0);
  input_data_size_ = bottom[1]->shape(1);
  
  //channels:the length of the hidden 
  channels_ = bottom[0]->shape(1);         //[1,250] [300,500]    
  M_ = num_;                //300
  N_ = channels_;           //1000
  K_ = input_data_size_;    //500
  num_hidden = bottom[0]->shape(1);      //250
  //this->blobs_.resize(2);
  //printf("hello word2");
  
  /*vector<int>weight_shape1(2);
  weight_shape1[0] = channels_;         
  weight_shape1[1] = num_hidden;    
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape1));   //Wa    250x250
  
  vector<int>weight_shape2(2);
  weight_shape2[0] = channels_;       
  weight_shape2[1] = input_data_size_;
  this->blobs_[1].reset(new Blob<Dtype>(weight_shape2));   //Ua   1000x500
  
  
 
  
  
  //fill the weight
  shared_ptr<Filler<Dtype> > lstm_hidden_weight_filler(GetFiller<Dtype>(
      indtanh_param.lstm_hidden_weight_filler()));
  lstm_hidden_weight_filler->Fill(this->blobs_[0].get());  //1000x250

  shared_ptr<Filler<Dtype> > attention_weight_filler(GetFiller<Dtype>(
  	indtanh_param.attention_weight_filler()));
  attention_weight_filler->Fill(this->blobs_[1].get());    //1000x500
  */

  
  lstm_hidden_data_buffer_.reset(new Blob<Dtype>());
  this->buffers_.push_back(lstm_hidden_data_buffer_);
  attention_data_buffer_.reset(new Blob<Dtype>());
  this->buffers_.push_back(attention_data_buffer_);
  context_buffer_.reset(new Blob<Dtype>());
  this->buffers_.push_back(context_buffer_);
  
  
 // indtan_diff_buffer_.reset(new Blob<Dtype>());
 // this->buffers_.push_back(indtan_diff_buffer_);
 // lstm_hidden_diff_buffer_.reset(new Blob<Dtype>()); 
 // this->buffers_.push_back(lstm_hidden_diff_buffer_);
 // attention_data_diff_buffer_.reset(new Blob<Dtype>());   
 // this->buffers_.push_back(attention_data_diff_buffer_);        
  context_diff_buffer_.reset(new Blob<Dtype>());
  this->buffers_.push_back(context_diff_buffer_);
  dldg_diff_buffer_.reset(new Blob<Dtype>());
  hidden_data_tmp_buffer_.reset(new Blob<Dtype>());
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  
}


 //300x500
template <typename Dtype>
void IndtanhLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK((this->layer_param_.bottom_size() == 2
      || this->layer_param_.bottom_size() == 0))
      << "indtanh must have a data and cell bottom";
  CHECK((this->layer_param_.top_size() == 1
      || this->layer_param_.top_size() == 0))
      << "indtanh must have a data and cell top";
  
  //lstm_hidden_diff_buffer_->Reshape(channels,input_data_size_),1,1);     //Wa:1000x500  
  //attention_diff_buffer_->Reshape(num_,channels,1,1);                  //Ua:300x500
  //context_data_diff_buffer_->Reshape(channels,num_,1,1);              //1000x300
  //lstm_hidden_diff_buffer_->Reshape(1,num_output,1,1);
  //attention_data_diff_buffer_->Reshape(num_, input_data_size_,1,1); 
 
  attention_data_buffer_->Reshape(num_, input_data_size_,1,1);
  lstm_hidden_data_buffer_->Reshape(1,num_hidden,1,1);
  context_buffer_->Reshape(channels_,num_,1,1);
  context_diff_buffer_->Reshape(channels_,num_,1,1);                  //1000x300
  dldg_diff_buffer_->Reshape(num_, num_hidden,1,1);
  hidden_data_tmp_buffer_->Reshape(num_,channels_,1,1);
  
  vector<int> shape;
  shape.push_back(channels_);
  shape.push_back(num_);
  top[0]->Reshape(shape);   //out_put 1000x300
}

template <typename Dtype>
void IndtanhLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* prev_state_data = bottom[0]->cpu_data();     //s(i-1):1x250    
  const Dtype* input_data = bottom[1]->cpu_data();          //H: 300x500
 
  
  //get the weight 
  const Dtype* lstm_hidden_weight = this->blobs_[0]->cpu_data();    
  printf("hello word2"); 
  const Dtype* attention_weight = this->blobs_[1]->cpu_data();
   printf("hello word2");
  //size: topdata 1000x300
  Dtype* indtanh_data = top[0]->mutable_cpu_data();
  
  //new data(mutable)
  Dtype* lstm_hidden_data = lstm_hidden_data_buffer_->mutable_cpu_data();
  Dtype*  attention_data = attention_data_buffer_->mutable_cpu_data();
  Dtype*  context_data = context_buffer_->mutable_cpu_data();
 
  int M_1 = 1;
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, M_1,num_hidden,     //1000, 1, 250
    (Dtype)1., lstm_hidden_weight, prev_state_data,
    (Dtype)0., lstm_hidden_data);                          //[250,250]X[1,250] = [250X1]
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, M_, K_,  //[1000,500]X[300,500] = [1000x300]
    (Dtype)1., attention_weight, input_data,
    (Dtype)0., attention_data);               
  

  for (int n = 0; n < channels_; ++n) {                   //1000x1+1000x300
    for (int i = 0; i < num_; ++i) {
      const int idx ontext_diff_buffer_-= i + n * num_;
      
      context_data[idx] = lstm_hidden_data[n] + attention_data[idx];  //w:1000x250  s: 250x1  u:1000x500 h:500x300
      indtanh_data[idx] = tanh(context_data[idx]);    
      printf("indtan_data_forward: %f  ", indtanh_data[4]);
    }
  }
  //printf("  forward  context_data:%f\n",context_data[4]);
}



template <typename Dtype>
void IndtanhLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   //printf("helloback,indtan");
  /*for (int i = 0; i < 2; ++i) {
    caffe_set(bottom[i]->count(), Dtype(0),
      bottom[i]->mutable_cpu_diff());
  }
  for (int i = 0; i < 2; ++i) {
    caffe_set(this->blobs_[i]->count(), Dtype(0),
      this->blobs_[i]->mutable_cpu_diff());                   //Wa, Ua
  }*/

  
  const Dtype* lstm_hidden = lstm_hidden_data_buffer_->cpu_data();
  const Dtype* attention_data = attention_data_buffer_->cpu_data();
  const Dtype* context_data = context_buffer_->cpu_data();            //1000x300
  
  //Dtype* lstm_hidden_diff = lstm_hidden_diff_buffer_->mutable_cpu_data();
  //Dtype* attention_data_diff = attention_data_diff_buffer_->mutable_cpu_data();
  Dtype* context_diff = context_diff_buffer_->mutable_cpu_data();  
  
  
  const Dtype* indtan_diff = top[0]->cpu_diff();
  const Dtype* indtan_data = top[0]->cpu_data();
  
  printf(" ind_diff:%f \n",indtan_diff[4]);
  //printf("    inddata:%f \n",indtan_data[4]);
  //printf("  attention_data_buffer_:%f\n",attention_data[4]);
  //printf("  backward  context_data:%f\n",context_data[4]);
  
  //compute the lstm_hidden_diff/context_diff
  
  for (int n = 0; n< channels_; ++n) {
  	for (int i =0; i<num_; ++i) {
  		const int idx = i + n*num_;
  		context_diff[idx] = tanh_diff(indtan_data[idx]);
  	}
  }
  
  //get the weight 
  const Dtype* lstm_hidden_weight = this->blobs_[0]->cpu_data();     
  const Dtype* attention_weight = this->blobs_[1]->cpu_data();
  
  Dtype* lstm_hidden_weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* attention_data_weight_diff = this->blobs_[1]->mutable_cpu_diff();
  

  Dtype* prev_state_tot_diff = dldg_diff_buffer_->mutable_cpu_data();
  Dtype* prev_state_diff = bottom[0]->mutable_cpu_diff();     //1X1000    
  Dtype* input_diff = bottom[1]->mutable_cpu_diff();          //300x500
  
  //Dtype* indtanh = indtanh  1000x300
  //Dtype* indtan_diff = top[0]->cpu_diff();     //u:1000x500 w:1000x250 h:300x250  s:1x250
  
  
  
  printf("context_diff %f",context_diff[2]);
  for (int n = 0; n< channels_; ++n) {           
  	for (int i =0; i<num_; ++i) {
  		const int idx = i + n*num_;
  		context_diff[idx] = context_diff[idx]*indtan_diff[idx];      
  	}
  }
  printf("context_diff %f",context_diff[2]);
  //compute the diff for all
  /*caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
   num_hidden, num_,  channels_ ,
    (Dtype)1.,lstm_hidden_weight,context_diff,  //context_diff:1000x300  lstm_hidden_weight: 1000x250  
    (Dtype)0., prev_state_tot_diff); */                  //250x300
   //printf("%d  ",prev_state_tot_diff+2);
   printf("prev_state_tot_diff %f",prev_state_tot_diff[2]);
 /*
   for (int i = 0;i<num_; ++i){
     for( int j=0;j<num_hidden; ++j){
        const int idx = j + i*num_hidden;
  		prev_state_diff[j] += prev_state_tot_diff[idx];    //1x250
  	}
  }*/
   
   
  //changed at 16:48 
  const Dtype* prev_state_data = bottom[0]->cpu_data();   //[1,250]
  
  Dtype* tmp_data = hidden_data_tmp_buffer_->mutable_cpu_data();
  for (int i = 0;i<num_;++i){
  	for(int j = 0;j<channels_;++j){
  	    const int idx = j + i*num_hidden;
  		tmp_data[idx] = prev_state_data[j];      
  	}
  }
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    channels_, channels_ , num_,
    (Dtype)1., context_diff,tmp_data,          //250x300, 300x250
    (Dtype)0., lstm_hidden_weight_diff);    //250x250
    
    
    //compute Ua_diff 250x500
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    channels_, input_data_size_, num_,
    (Dtype)1., context_diff, bottom[1]->cpu_data(),             //context_diff :1000x300 input_data:300x500
    (Dtype)0., attention_data_weight_diff);                     //g(t)_weight   1000x500          
    //compute H_diff 300x500
    /*caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    num_, input_data_size_, channels_,
    (Dtype)1., context_diff, attention_weight,   //context_diff:250x300 attention_weight:250x500
    (Dtype)0., input_diff);                     //300x500*/
    printf("    indtan_diff bottom_diff: %f  \n",input_ [2]);
   
}

#ifdef CPU_ONLY
STUB_GPU(IndtanhLayer);
#endif

INSTANTIATE_CLASS(IndtanhLayer);
REGISTER_LAYER_CLASS(Indtanh);

}  // namespace caffe