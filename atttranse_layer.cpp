#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AtttranseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(2);
  top_shape[0] = bottom[0]->shape(0);
  top_shape[1] = bottom[0]->shape(1);
  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count());
}

template <typename Dtype>
void AtttranseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
 // vector<int> top_shape = bottom[0]->shape();
 // const int dim = top_shape[1];
 const int dim = bottom[0]->shape(1);
  int height = 15;
  int weight = 20;
 
  
  //get the time flag 
  int step = this->layer_param_.atttranse_param().step();
  //copy data from bottom to top   300x1024-->300x1024
  for(int i=0; i<height; ++i){
    for(int j=0; j<weight;++j){
    	for(int c=0;c<dim; ++c){
    		if(step == 0){      //a9
    		 	if(i!=0 && j!=0)
    		 		top_data[(i*weight+j)*dim+c] = bottom_data[((i-1)*weight+(j-1))*dim+c];
    		 	else
    		 		top_data[(i*weight+j)*dim+c] =0;
			}
			else if(step ==1){   //a8
				if(j!= 0)
					top_data[(i*weight+j)*dim+c] = bottom_data[((i)*weight+(j-1))*dim+c];	
				else
    		 		top_data[(i*weight+j)*dim+c] =0;
			}
			else if(step == 2){   //a6
				if(i!=height-1 || j!=0)
					top_data[(i*weight+j)*dim+c] = bottom_data[((i+1)*weight+(j-1))*dim+c];
				else
    		 		top_data[(i*weight+j)*dim+c] =0;
			}
			else if(step == 3){
				if(i!=height-1)
					top_data[(i*weight+j)*dim+c] = bottom_data[((i+1)*weight+(j))*dim+c];
				else
    		 		top_data[(i*weight+j)*dim+c] =0;
			}
			else if(step == 4){
				if(i!=height-1 || j!=weight-1)
					top_data[(i*weight+j)*dim+c] = bottom_data[((i+1)*weight+(j+1))*dim+c];
				else
    		 		top_data[(i*weight+j)*dim+c] =0;
			}
			else if(step == 5){
				if(j!=weight)
					top_data[(i*weight+j)*dim+c] = bottom_data[((i-1)*weight+(j+1))*dim+c];
				else
    		 		top_data[(i*weight+j)*dim+c] =0;
			}
			else if(step == 6){
				if(j!=0 || j!=weight-1)
					top_data[(i*weight+j)*dim+c] = bottom_data[((i-1)*weight+(j+1))*dim+c];
				else
    		 		top_data[(i*weight+j)*dim+c] =0;
			}
			else if(step == 7){
				if(i!=0)
					top_data[(i*weight+j)*dim+c] = bottom_data[((i-1)*weight+(j))*dim+c];
				else
    		 		top_data[(i*weight+j)*dim+c] =0;
			}
			else
				top_data[(i*weight+j)*dim+c] = bottom_data[((i)*weight+(j))*dim+c];
    	}
  	}
  }

}

template <typename Dtype>
void AtttranseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  //vector<int> bottom_shape = bottom[0]->shape();
  
//shape(0)
//shape(1)
 // const int dim = bottom_shape[1];
const int dim = bottom[0]->shape(1);
  int height = 15;
  int weight = 20;
  
  
  
  //get the time flag 
  int step = this->layer_param_.atttranse_param().step();
  for(int i=0; i<height; ++i){
    for(int j=0; j<weight;++j){
    	for(int c=0;c<dim; ++c){
    		/*if(step == 0){      //a9
    		 	if(i!=0 && j!=0)
    		 		bottom_diff[((i-1)*weight+(j-1))*dim+c]=top_diff[(i*weight+j)*dim+c];
    		 	//else
    		 	//	bottom_diff[((i-1)*weight+(j-1))*dim+c] =0;
			}
			else if(step ==1){   //a8
				if(j!= 0)
					bottom_diff[((i)*weight+(j-1))*dim+c]=top_diff[(i*weight+j)*dim+c];
    		 	//else
    		 	//	bottom_diff[((i)*weight+(j-1))*dim+c] =0;
			}
			else if(step == 2){   //a6
				if(i!=height-1 || j!=0)
					bottom_diff[((i+1)*weight+(j-1))*dim+c]=top_diff[(i*weight+j)*dim+c];
    		 	//else
    		 	//	bottom_diff[((i+1)*weight+(j-1))*dim+c] =0;
			}
			else if(step == 3){
			    
				if(i!=height-1)
				     bottom_diff[((i+1)*weight+(j))*dim+c]=top_diff[(i*weight+j)*dim+c];
					//bottom_diff[((i+1)*weight+(j))*dim+c]=top_diff[(i*weight+j)*dim+c];
    		 	//else
    		 	//	bottom_diff[((i+1)*weight+(j))*dim+c] =0;
			}
			else if(step == 4){
			    //printf("atttanse_layer: ");
				if(i!=height-1 || j!=weight-1)
					bottom_diff[((i+1)*weight+(j+1))*dim+c]=top_diff[(i*weight+j)*dim+c];
					
    		 	//else
    		 	//	bottom_diff[((i+1)*weight+(j+1))*dim+c] =0;
    		 	//printf("%f ",bottom_diff[((i)*weight+(j))*dim+c]);
			}
			else if(step == 5){
				if(j!=weight)
					bottom_diff[((i-1)*weight+(j+1))*dim+c]=top_diff[(i*weight+j)*dim+c];
    		 	//else
    		 	//	bottom_diff[((i-1)*weight+(j+1))*dim+c] =0;
			}
			else if(step == 6){
				if(j!=0 || j!=weight-1)
					bottom_diff[((i-1)*weight+(j+1))*dim+c]=top_diff[(i*weight+j)*dim+c];
    		 	//else
    		 	//	bottom_diff[((i-1)*weight+(j+1))*dim+c] =0;
			}
			else if(step == 7){
				if(i!=0)
					bottom_diff[((i-1)*weight+(j))*dim+c]=top_diff[(i*weight+j)*dim+c];
    		 	//else
    		 	//	bottom_diff[((i-1)*weight+(j))*dim+c] =0;
			}*/
			if(step == 8)
				bottom_diff[((i)*weight+(j))*dim+c]=top_diff[(i*weight+j)*dim+c];
				//printf("%f ",bottom_diff[((i)*weight+(j))*dim+c]);
    	}
  	}
  }
  
  
}


#ifdef CPU_ONLY
STUB_GPU(AtttranseLayer);
#endif

INSTANTIATE_CLASS(AtttranseLayer);
REGISTER_LAYER_CLASS(Atttranse);

}  // namespace caffe
