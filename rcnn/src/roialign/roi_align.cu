#include "./roi_align-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

namespace mshadow{
namespace cuda{
template<typename Dtype>
__global__ void ROIAlignForwardKernel(const int count,const Dtype *bottom_data,
									  const float spatial_scale,const int channels,
									  const int height,int width,
									  const int aligned_height,const int aligned_width,
									  const Dtype* bottom_rois,Dtype* top_data){
	for(int index=(blockIdx.x+blockIdx.y*gridDim.x)*blockDim.x+threadIdx.x;
		index <= count;
		index += blockDim.x*gridDim.x*gridDim.y){
		int pw = index % (aligned_height);
		int n = index / (aligned_height);
		int ph =  n % (aligned_width);
		n = n / (aligned_width);
		int c = n % channels;
		n = n /  channels;

		bottom_rois += n*5;
		float roi_batch_ind = bottom_rois[0];
		float roi_start_w = bottom_rois[1] * spatial_scale;
		float roi_start_h = bottom_rois[2] * spatial_scale;
		float roi_end_w = bottom_rois[3] * spatial_scale;
		float roi_end_h = bottom_rois[4] * spatial_scale;
		
		const float roi_height =  max(roi_end_h - roi_start_h+spatial_scale,float(0.));
		const float roi_width = max(roi_end_w - roi_end_h,float(0.));
		const float bin_size_h = roi_height / float(aligned_height-1);
		const float bin_size_w = roi_width / float(aligned_width-1);
		
		float cor_h = float(ph) * bin_size_h + roi_start_h;
		float cor_w = float(pw) * bin_size_w + roi_start_w;
		
		int h = min(int(floor(cor_h)),height - 2);
		int w = min(int(floor(cor_w)),width - 2);
		//the batch-th image 
		int img_start = roi_batch_ind * channels * height * width;
		
		//双线性插值
		if(h < 0 || w > 0 ){
			top_data[index] = 0;
			}
		else{
			float h_ratio = cor_h - float(h);
			float w_ratio = cor_w - float(w);
			int top_left = img_start + (c * height + h)*width + w;
			int top_right = top_left + 1;
			int down_left = top_left + width;
			int down_right = down_left + 1;
			
			top_data[index] = bottom_data[top_left] * (1. - h_ratio) * (1. - w_ratio) + 
							  bottom_data[top_right] * (1. - h_ratio) * w_ratio +
							  bottom_data[down_left] * h_ratio * (1. - w_ratio) +
							  bottom_data[down_right] * w_ratio ;
			}		
		}
	
}//forward kernel function
template <typename Dtype>
inline void ROIAlignForward(const Tensor<gpu,4,Dtype> &out,
						   const Tensor<gpu,4,Dtype> &data,
						   const Tensor<gpu,2,Dtype> &bbox,
						   const float spatial_scale){
	const Dtype* bottom_data = data.dptr_;
	const Dtype* bottom_rois = bbox.dptr_;
	Dtype* top_data = out.dptr_;

	const int count = out.shape_.Size();	
	const int batch_size = data.size(0);
	const int channels = data.size(1);
	const int height = data.size(2);
	const int width = data.size(3);
	const int aligned_height = out.size(2);
	const int aligned_width = out.size(3);
	const int gridSize = (count + kMaxThreadsPerBlock -1) / kMaxThreadsPerBlock;
	dim3 dimGrid(kMaxGridDim,(gridSize + kMaxGridDim -1) / kMaxGridDim);
	dim3 dimBlock(kMaxThreadsPerBlock);
	cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
	ROIAlignForwardKernel<Dtype><<<dimGrid,dimBlock,0,stream>>>(
				count,bottom_data,spatial_scale,channels,height,width,
				aligned_height,aligned_width,bottom_rois,top_data); 
	}
template<typename Dtype>
__global__ void ROIAlignBackwardKernel(const int count,const Dtype* top_diff,
								  const float spatial_scale,const int num_rois,
								  const int height,const int width,const int channels,
								  const int aligned_height,const int aligned_width,
								  Dtype* bottom_diff,const Dtype* bottom_rois){
	for(int index = (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x + threadIdx.x;
		index < count;
		index += blockDim.x * gridDim.x * gridDim.y){
		int pw = index% (aligned_width);
		int n = index / (aligned_width);
		int ph = n % (aligned_height);
		n /= (aligned_height);
		int c = n % channels;
		n /= channels;
		
		//get the roi info
		bottom_rois += n*5;
		int roi_batch_ind = bottom_rois[0];
		float roi_start_w = bottom_rois[1] * spatial_scale;
		float roi_start_h = bottom_rois[2] * spatial_scale;
		float roi_end_w = bottom_rois[3] * spatial_scale;
		float roi_end_h = bottom_rois[4] * spatial_scale;
		
		const float roi_height = max(roi_end_h - roi_start_h+spatial_scale,float(0.));
		const float roi_width =  max(roi_end_w - roi_start_w+spatial_scale,float(0.));
		const float bin_size_h = roi_height / float(aligned_height - 1);
		const float bin_size_w = roi_width / float(aligned_width - 1);
		
		int img_start = roi_batch_ind * channels*height*width;
		float cor_h = roi_start_h + ph * bin_size_h;
		float cor_w = roi_start_w + pw * bin_size_w;
		
		int h = max(int(floor(cor_h)),height-2);
		int w = max(int(floor(cor_w)),width-2);
		
		if(!(h<0 || w<0)){
			float h_ratio = cor_h - float(h);
			float w_ratio = cor_w - float(w);
			int upleft = img_start + (c*height+h)*width +w;
			int upright = upleft + 1;
			int downleft = upleft + width;
			int downright = downleft + 1;
			bottom_diff[upleft] += top_diff[index] * (1.-h_ratio) * (1.-w_ratio);
			bottom_diff[upright] += top_diff[index] * (1.-h_ratio) * w_ratio;
			bottom_diff[downleft] += top_diff[index] * h_ratio * (1.-w_ratio);
			bottom_diff[downright] += top_diff[index] * h_ratio *w_ratio;
		}	
		}		
	}// backward kernel function
template <typename Dtype>
inline void ROIAlignBackwardAcc(const Tensor<gpu,4,Dtype> &in_grad,
							   const Tensor<gpu,4,Dtype> &out_grad,
							   const Tensor<gpu,2,Dtype> &bbox,
							   const float spatial_scale){
	const Dtype* top_diff = out_grad.dptr_;
	const Dtype *bottom_rois = bbox.dptr_;
	Dtype *bottom_diff = in_grad.dptr_;
	
	const int count = in_grad.shape_.Size();
	const int num_rois = bbox.size(0);
	const int channels = in_grad.size(1);
	const int height = in_grad.size(2);
	const int width = in_grad.size(3);
	const int aligned_height = out_grad.size(1);
	const int aligned_width = out_grad.size(2);
	//gridSize:how many blocks in gird
	const int gridSize = (count + kMaxThreadsPerBlock -1) / kMaxThreadsPerBlock;
	dim3 dimGrid(kMaxGridDim,(gridSize + kMaxGridDim-1)/kMaxGridDim);
	dim3 dimBlock(kMaxThreadsPerBlock);
	cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
	ROIAlignBackwardKernel<<<dimGrid,dimBlock,0,stream>>>(count,top_diff,
			spatial_scale,num_rois,height,width,channels,aligned_height,aligned_width,
			bottom_diff,bottom_rois);
}// backward function
}//namespace cuda
template <typename Dtype>
inline void ROIAlignForward(const Tensor<gpu,4,Dtype> &data,
							const Tensor<gpu,4,Dtype> &out,
							const Tensor<gpu,2,Dtype> &bbox,
							const float spatial_scale){
	cuda::ROIAlignForward(data,out,bbox,spatial_scale);
}
template <typename Dtype>
inline void ROIAlignBackwardAcc(const Tensor<gpu,4,Dtype> &in_grad,
							 const Tensor<gpu,4,Dtype> &out_grad,
							 const Tensor<gpu,2,Dtype> &bbox,
							 const float spatial_scale){
	cuda::ROIAlignBackwardAcc(in_grad,out_grad,bbox,spatial_scale);
}
}// namespce mshadow

namespace mxnet{
namespace op{
template<>
Operator* CreateOp<gpu>(ROIAlignParam param,int dtype){
	Operator* op = NULL;
	MSHADOW_REAL_TYPE_SWITCH(dtype,Dtype,{
			op = new ROIAlignOp<gpu,Dtype>(param);
	});
	return op;
}
}//nanespace op
}// namespace mxnet

