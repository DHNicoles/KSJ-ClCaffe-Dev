/************************************************************************/
/* detect and track process .                                           */
/************************************************************************/

#ifndef __KERNEL_ALG_H__
#define __KERNEL_ALG_H__

#include "../utils/util.h"
#include "../cnn_multi_tracker/cnn_multi_tracker.h"
namespace ice
{
	////////////////////////////////////
	//local def
	////////////////////////////////////
	extern double l_iou_thresh;

	////////////////////////////////////
	class Detector;
	class KernelAlg :public Singleton<KernelAlg>
	{
		
	public:
		KernelAlg();
		~KernelAlg();
		void OnInit();
		void OnDestroy();
		void OnUpdate(cv::Mat& frame, size_t offset);
		void SetBound(cv::Size, cv::Rect);//detector_win, trackor_win
		bool Inited;

	private:
		void Match(std::vector<cv::Rect>& detect_boxes, cv::Mat& frame);
		void Match_ex(std::vector<cv::Rect>& detect_boxes, cv::Mat& frame);
	private:
		Detector* detector_ptr_;
		CNNMultiTracker* cnn_multi_trackers_ptr_;
	private:
	};

	
}

#endif // __KERNEL_ALG_H__
