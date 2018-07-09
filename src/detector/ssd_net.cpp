/*************************************************************************
  > File Name: ../src/ssd_net.cpp
  > Author: cheguangfu
  > Mail: cheguangfu1@jd.com 
  > Created Time: 2017年09月11日 星期一 11时33分10秒
 ************************************************************************/
#include "caffe/layers/memory_data_layer.hpp"
#include "detector/ssd_net.h"
#include "detector/intel_detector.hpp"

namespace ice
{
	SSDNet::SSDNet()
	{
		mp_detector_ = NULL;
	}

	SSDNet::~SSDNet()
	{
		OnDestroy();
	}
	void SSDNet::OnInit(const std::string& model_file, const std::string& weights_file, bool is_big_model)
	{
        if(is_big_model)
        {
            Caffe::set_mode(Caffe::GPU);
            Caffe::SetDevice(0);
            /* Load the network. */
            net_.reset(new Net<float>(model_file, TEST, Caffe::GetDefaultDevice()));
            net_->CopyTrainedLayersFrom(weights_file);

            //CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
            //CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";


            Blob<float>* input_layer = net_->input_blobs()[0];
            num_channels_ = input_layer->channels();
            //CHECK(num_channels_ == 3 || num_channels_ == 1)
            //	<< "Input layer should have 1 or 3 channels.";
            input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
        }
        else
        {
            mp_detector_ = new IntelDetector(model_file, weights_file, 1, false);	//init with a default batch, and the batch can be changed
        }
    }
    void SSDNet::OnDestroy()
    {
        if(mp_detector_)
            delete mp_detector_;
        mp_detector_ = NULL;
    }

    void SSDNet::Detect(std::vector<cv::Mat>& img_v, std::vector<std::vector<cv::Rect> >& head_box_v, std::vector<std::vector<float> >& head_score_v, float detect_thresh)
    {
        //LOG(INFO) << "Insert img_vec size: " << img_v.size();
        //static Stopwatch T("forward");
        //T.Reset();   T.Start();
        for(int i=0;i<img_v.size();++i){
            mp_detector_->InsertImage(img_v[i],i,img_v.size());
        }
        //here can async 
        int curbatch;
        if((curbatch=mp_detector_->TryDetect())<=0){
            LOG(INFO) << "no data to detect \n" ;
            return;
        }		
        vector<IntelDetector::Result> objects(curbatch);
        mp_detector_->Detect(objects);

        //T.Stop();
        //LOG(INFO) << "model forward cost: " << T.GetTime();

        head_box_v.resize(objects.size());
        head_score_v.resize(objects.size());

        for(int k=0;k<objects.size();k++){
            for(int i=0;i<objects[k].boxs.size();i++){
                if(objects[k].boxs[i].confidence>detect_thresh && objects[k].boxs[i].classid==1){
                    //LOG(INFO) <<"[xmin,ymin,xmax,ymax]: [" << objects[k].boxs[i].left << ","  << objects[k].boxs[i].top << "," << objects[k].boxs[i].right << "," << objects[k].boxs[i].bottom <<"]";
                    //cv::Rect box =  cv::Rect(cv::Point(result[3] * img_v[label].cols, result[4] * img_v[label].rows),cv::Point(result[5] * img_v[label].cols, result[6] * img_v[label].rows));
                    cv::Rect box =  cv::Rect(cv::Point(objects[k].boxs[i].left, objects[k].boxs[i].top),cv::Point(objects[k].boxs[i].right, objects[k].boxs[i].bottom));
                    head_box_v[k].push_back(box);
                    //这里我就搞不明白了,  为啥要乘以w.h 也就是img_v[label].cols, img_v[label].row
                    head_score_v[k].push_back(objects[k].boxs[i].confidence);
                }
            }			
        }
    }
	void SSDNet::Detect(const cv::Mat& img, std::vector<cv::Rect>& head_box_v, std::vector<float>& head_score_v, float detect_thresh)
	{
        LOG(INFO) << "----img size---" << img.size();
		Blob<float>* input_layer = net_->input_blobs()[0];
		input_layer->Reshape(1, num_channels_,
				input_geometry_.height, input_geometry_.width);
		/* Forward dimension change to all layers. */
		net_->Reshape();

		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels);

		Preprocess(img, &input_channels);

		net_->Forward();

		/* Copy the output layer to a std::vector */
		Blob<float>* result_blob = net_->output_blobs()[0];
		const float* result = result_blob->cpu_data();
		const int num_det = result_blob->height();
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		for (int k = 0; k < num_det; ++k) 
		{
			if (result[0] == -1) {
				// Skip invalid detection.
				result += 7;
				continue;
			}
			if(result[1] == 1 && result[2] > detect_thresh)//head and thresh
			{
				cv::Rect box =  cv::Rect(cv::Point(result[3] * img.cols, result[4] * img.rows),cv::Point(result[5] * img.cols, result[6] * img.rows));
				head_box_v.push_back(box);
				head_score_v.push_back(result[2]);
			}
			result += 7;
		}
	}
	/* Wrap the input layer of the network in separate cv::Mat objects
	 * (one per channel). This way we save one memcpy operation and we
	 * don't need to rely on cudaMemcpy2D. The last preprocessing
	 * operation will write the separate channels directly to the input
	 * layer. */
	void SSDNet::WrapInputLayer(std::vector<cv::Mat>* input_channels) 
	{
		Blob<float>* input_layer = net_->input_blobs()[0];

		int width = input_layer->width();
		int height = input_layer->height();
		float* input_data = input_layer->mutable_cpu_data();
		for (int i = 0; i < input_layer->channels(); ++i) {
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}
	}

	void SSDNet::Preprocess(const cv::Mat& img,
			std::vector<cv::Mat>* input_channels) {
		/* Convert the input image to the input image format of the network. */
		cv::Mat sample;
		if (img.channels() == 3 && num_channels_ == 1)
			cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
		else if (img.channels() == 4 && num_channels_ == 1)
			cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
		else if (img.channels() == 4 && num_channels_ == 3)
			cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
		else if (img.channels() == 1 && num_channels_ == 3)
			cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
		else
			sample = img;

		cv::Mat sample_resized;
		if (sample.size() != input_geometry_)
			cv::resize(sample, sample_resized, input_geometry_);
		else
			sample_resized = sample;

		cv::Mat sample_float;
		if (num_channels_ == 3)
			sample_resized.convertTo(sample_float, CV_32FC3);
		else
			sample_resized.convertTo(sample_float, CV_32FC1);

		cv::Scalar mean(127.5, 127.5, 127.5);
		sample_float -= mean;
		cv::Mat sample_normalized;
		cv::divide(sample_float, 127.0, sample_normalized);

		/* This operation will write the separate BGR planes directly to the
		 * input layer of the network because it is wrapped by the cv::Mat
		 * objects in input_channels. */
		cv::split(sample_normalized, *input_channels);

		CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
				== net_->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";
	}


}
