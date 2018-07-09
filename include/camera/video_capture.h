#ifndef video_capture_h__
#define video_capture_h__
#include "utils/util.h" 
#include <cstring>
#include "camera/KSJ_inc/KSJApi.h"
#include <unistd.h>
#define CALL_BACK
namespace ice
{
	class VideoCapture
	{
		
	private:
		/************************************************************************/
		/* static global callback function and var                                       */
		/************************************************************************/
	public:
		VideoCapture();
		~VideoCapture();
		int OnInit();
        int Start();
        int Stop();
        int BufSize() { return mat_buf_.size();}
		void OnDestroy();
		cv::Mat Query();
		VideoCapture& operator>>(cv::Mat& src);
	private:
		static void* CapThread(void* param);
	private:
		int KSJ_SetCamsParam();
		unsigned char* buf_;
		IplImage * p_image_; 
		int camera_index_;
        volatile bool cap_continue_;
        //std::deque<cv::Mat> mat_buf_;
        std::list<cv::Mat> mat_buf_;
        pthread_mutex_t buf_lock_;
		const int mat_buf_max_size_ = 100;
	};


}
#endif // video_capture_h__
