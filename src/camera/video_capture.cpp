#include "camera/video_capture.h"

namespace ice
{
    ////////////////////////////////////////////////
    //local var	
    //const int l_width = 3096;
    int g_startx = 348;
    int g_starty = 40;
    int g_sizex = 2400;
    int g_sizey = 2000;
    float g_exp = 20;
    int g_gain = 100;
    int l_width = 2400;
    int l_height = 2000;
    extern cv::RNG l_rng;
    ////////////////////////////////////////////////

    VideoCapture::VideoCapture()
    {
        buf_ = NULL;
        buf_lock_ = PTHREAD_MUTEX_INITIALIZER;
    }

    VideoCapture::~VideoCapture()
    {
        OnDestroy();
    }

    int VideoCapture::OnInit()
    {
        int ret = KSJ_Init();
        int n_cam_count = KSJ_DeviceGetCount();
        if(n_cam_count <=0 )
        {
            LOG(FATAL) << "No KSJ device was found!";
            return RETCODES_INIT_ERROR;
        }
        else if(n_cam_count != 1)
        {
            LOG(INFO) << "There are (" << n_cam_count << ") KSJ device found! But only device 0 will be used.";
        }
        camera_index_ = 0;
        KSJ_SetCamsParam();
        int n_size = l_width * l_height * 3;
        buf_ = (unsigned char*)malloc(n_size);
        p_image_ = cvCreateImage(cvSize(l_width, l_height), IPL_DEPTH_8U, 3);
        p_image_->imageData = (char*)buf_;
        return RETCODES_SUCCESS;
    }
    int VideoCapture::Start()
    {
		pthread_t id;
		cap_continue_ = true;
		LOG(INFO) << "capture thread starting...";
		int ret = pthread_create(&id, NULL, CapThread, this);
		if (ret) {
			LOG(FATAL) << "Create pthread error!";
			return 0;
		}
		LOG(INFO) << "post thread start success!";
        return 0;
    }

void* VideoCapture::CapThread(void* param)
	{
		pthread_detach(pthread_self());
		VideoCapture* this_ptr = (VideoCapture*)param;
		//std::deque<cv::Mat>& mat_buf = this_ptr->mat_buf_;
		std::list<cv::Mat>& mat_buf = this_ptr->mat_buf_;
        pthread_mutex_t& buf_lock = this_ptr->buf_lock_;
		Stopwatch T0;
		while (this_ptr->cap_continue_)
		{
			T0.Reset();	T0.Start();

			/////////////////////////
			//capture runing  ///////
			/////////////////////////
			
    		cv::Mat frame = this_ptr->Query();
			pthread_mutex_lock(&buf_lock);
			if(mat_buf.size() >= this_ptr->mat_buf_max_size_)
			{
                //rand del one
                int idx = l_rng.uniform(0, this_ptr->mat_buf_max_size_);
                LOG(INFO) << "rand delete index: " << idx;
                std::list<cv::Mat>::iterator bitr =  mat_buf.begin();
                std::advance(bitr, idx);
                mat_buf.erase(bitr);
            }
            mat_buf.push_back(frame);
            LOG(INFO) << "[Cap Thread] mat_buf size:  " << mat_buf.size();
            pthread_mutex_unlock(&buf_lock);
			T0.Stop();
			LOG(INFO) << "[Cap Thread] cost " << T0.GetTime();
		}
		LOG(INFO) << "[Cap Thread] thread exit.";
		return nullptr;
	}

    int VideoCapture::Stop()
    {
		cap_continue_ = false;
	}
    int VideoCapture::KSJ_SetCamsParam()
    {
        int nRet = 0;
        int nColStart = 0;
        int nRowStart = 0;
        int ggx;
        int nColSize = 0;
        int nRowSize = 0;
        KSJ_ADDRESSMODE ColAddressMode;
        KSJ_ADDRESSMODE RowAddressMode;

        KSJ_CaptureGetDefaultFieldOfView(camera_index_,(int*)&nColStart,(int*)&nRowStart,(int *)&nColSize,(int *)&nRowSize,&ColAddressMode,&RowAddressMode);

        KSJ_CaptureSetFieldOfView(camera_index_,g_startx,g_starty,g_sizex,g_sizey,KSJ_SKIP2,KSJ_SKIP2);
        //		KSJ_CaptureSetFieldOfView(camera_index_,g_startx,g_starty,g_sizex,g_sizey, KSJ_SKIPNONE, KSJ_SKIPNONE);



        KSJ_CaptureGetFieldOfView(camera_index_,&g_startx,&g_starty,&g_sizex,&g_sizey,&ColAddressMode,&RowAddressMode);
        LOG(INFO) << "RowAddressMode = " << RowAddressMode;
        LOG(INFO) << "ColAddressMode = " << ColAddressMode;
        LOG(INFO) << "nRowStart = " << nRowStart;
        LOG(INFO) << "nColStart = " << nColStart;
        LOG(INFO) << "nColSize  = " << nColSize;
        LOG(INFO) << "nRowSize  = " << nRowSize;

        KSJ_CaptureGetSize(camera_index_,&l_width,&l_height);
        LOG(INFO) << "nColStart = " << nColStart;

        LOG(INFO) << "l_width = " << l_width;
        LOG(INFO) << "l_height = " << l_height;
        KSJ_BayerSetMode(camera_index_, KSJ_RGGB_BGR24_FLIP);

        int nlines = 0;
        LOG(INFO) << "ExposureTime = " << g_exp;
        KSJ_ExposureTimeSet(camera_index_,g_exp);
        KSJ_WB_MODE wbmode = KSJ_HWB_PRESETTINGS;
        //		KSJ_WB_MODE wbmode = KSJ_HWB_AUTO_CONITNUOUS;
        LOG(INFO) << "KSJ_WhiteBalanceSet retcode: " << KSJ_WhiteBalanceSet(camera_index_,wbmode);

        KSJ_WhiteBalanceGet(camera_index_,&wbmode);
        KSJ_SetParam(camera_index_,KSJ_RED,g_gain);
        KSJ_SetParam(camera_index_,KSJ_GREEN,g_gain);
        KSJ_SetParam(camera_index_,KSJ_BLUE,g_gain);
        LOG(INFO) << "KSJ_WhiteBalanceGet " << wbmode;
        KSJ_SENSITIVITYMODE  smode  =  KSJ_HIGH;
        LOG(INFO) << "line " << __LINE__;
        KSJ_SensitivitySetMode(camera_index_,smode);
        LOG(INFO) << "line " << __LINE__;
        KSJ_ColorCorrectionSet(camera_index_, KSJ_HCCM_PRESETTINGS);
        LOG(INFO) << "line " << __LINE__;
        KSJ_LutSetEnable(camera_index_,1);
        LOG(INFO) << "line " << __LINE__;

        return 0;
    }
    void VideoCapture::OnDestroy()
    {
        if(buf_) free(buf_);
        buf_ = NULL;

        cvReleaseImage(&p_image_);
    }

    cv::Mat VideoCapture::Query()
    {
        static Stopwatch T_v("KSJ_timer");
        //T_v.Reset();    T_v.Start();	
        while(0 != KSJ_CaptureRgbData(camera_index_, buf_))
        {
            LOG(ERROR) <<"KSJ capture data once again, because bad capture exception at device "<< camera_index_;
            continue;
        }
        //T_v.Stop();
        //LOG(INFO) << "KSJ capture time:\t" << T_v.GetTime() << endl;
        cv::Mat frame = cv::cvarrToMat(p_image_);	
        //frame.convertTo(frame, frame.type(), 2.5, 15);
        return frame;
    }
    VideoCapture& VideoCapture::operator>>(cv::Mat& src)
    {
        while(mat_buf_.empty())
        {
            usleep(1000);
        }
		pthread_mutex_lock(&buf_lock_);
		src = mat_buf_.front();
		mat_buf_.pop_front();
		pthread_mutex_unlock(&buf_lock_);
		return *this;
    }
}
