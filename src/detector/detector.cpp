#include "detector/detector.h"
#include "detector/ssd_net.h"

namespace ice
{
	Detector::Detector()
		:ssd_global_net_ptr_(NULL), ssd_local_net_ptr_(NULL)
	{
	}

	Detector::~Detector()
	{
		Destroy();
	}

	void Detector::OnInit()
	{
		const int MAXBUFSIZE = 1024;
		char buf[ MAXBUFSIZE ];
		getcwd(buf, MAXBUFSIZE);
		LOG(INFO) << "pwd: " << buf;
		
		/////////////////////////////////////////////////
		//Use DETECTOR_TYPE_MTCNN or DETECTOR_TYPE_SSD
		//detector_type = DETECTOR_TYPE_MTCNN;
		detector_type = DETECTOR_TYPE_SSD;
		/////////////////////////////////////////////////

		//ssd-net init
		/////////////////////////////////////////////////
#if 0 
		std::string model = "../../resource/model/mobilessd/fused_MobileNetSSD_deploy.prototxt";
		std::string weights = "../../resource/model/mobilessd/fused_MobileNetSSD_deploy.caffemodel";
#else
		std::string model_g = "../../resource/model/50p_180309/fused_MobileNetSSD_deploy.prototxt";
		std::string weights_g = "../../resource/model/50p_180309/fused_MobileNetSSD_deploy.caffemodel";
		std::string model_l = "../../resource/model/trim_300_local/fused_MobileNetSSD_deploy.prototxt";
		std::string weights_l = "../../resource/model/trim_300_local/fused_MobileNetSSD_deploy.caffemodel";
#endif
		ssd_global_net_ptr_ = new SSDNet;
		ssd_global_net_ptr_->OnInit(model_g, weights_g, false);
		ssd_local_net_ptr_ = new SSDNet;
		ssd_local_net_ptr_->OnInit(model_l, weights_l, false);
	}

	void Detector::Destroy()
	{
		if (ssd_global_net_ptr_)
			delete ssd_global_net_ptr_;
		ssd_global_net_ptr_ = NULL;

		if (ssd_local_net_ptr_)
			delete ssd_local_net_ptr_;
		ssd_local_net_ptr_ = NULL;

	}

	void Detector::Detect(cv::Mat& src, std::vector<cv::Rect>& targets, std::vector<float>& score, DETECTOR_SCALE detect_scale)
	{
		if (src.empty())
		{
			LOG(ERROR) << "detector a empty frame. skip";
			return;
		}
        if(detect_scale == DETECTOR_SCALE_GLOBAL)
        {
            ssd_global_net_ptr_->Detect(src, targets, score, 0.5);
        }
        else if(detect_scale == DETECTOR_SCALE_LOCAL)
        {
            ssd_local_net_ptr_->Detect(src, targets, score, 0.1);
        }
	}
    void Detector::Detect(std::vector<cv::Mat>& src_v, std::vector<std::vector<cv::Rect> >& targets, std::vector<std::vector<float> >& score, DETECTOR_SCALE detect_scale)
	{
		if (src_v.empty())
		{
			LOG(ERROR) << "detector a empty frame. skip";
			return;
		}
        if(detect_scale == DETECTOR_SCALE_GLOBAL)
        {
            //support
            //LOG(INFO) << "---Use big model---";
            ssd_global_net_ptr_->Detect(src_v, targets, score, 0.5);
        }
        else if(detect_scale == DETECTOR_SCALE_LOCAL)
        {
            //LOG(INFO) << "---Use small model---";
            ssd_local_net_ptr_->Detect(src_v, targets, score, 0.1);
        }
	}
}
