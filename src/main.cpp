#include "game/logic.h"
#include "camera/video_capture.h"
#include "utils/util.h"
//#include <X11/Xlib.h>   
int main(int argc, char* argv[])
{
	//XInitThreads();
	////////////////////
	//parsing args
	////////////////////
	if (argc != 2)
	{
		LOG(ERROR) << "Usage : " << argv[0] << " your_cgf_file";
		return -1;
	}
	else
	{
		const char* cfg = argv[1];
		if (ice::parseCfg(cfg) != 0)
		{
			LOG(ERROR) << cfg << " file parse failed!";
			return -1;
		}
	}
	////////////////////
	//checking camera
	////////////////////
	ice::VideoCapture cap_ksj;
	cv::VideoCapture cap_usb("../../resource/video/0001.avi");
	if(ice::g_use_ksj)
	{
		if (ice::RETCODES_SUCCESS != cap_ksj.OnInit())
		{
			LOG(ERROR) << "Fatal : camera init failed";
			return -1;
		}
	}
	else
	{
		//cv::VideoCapture cap(1);
		if (!cap_usb.isOpened())
		{
			LOG(ERROR) << "Fatal : camera init failed";
			return -1;
		}
		cap_usb.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		cap_usb.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	}
	////////////////////
	//start args
	////////////////////
	ice::KernelAlg alg;


	cv::Mat frame;
	if(ice::g_use_ksj)
	{
        cap_ksj.Start();
		cap_ksj >> frame;
		//cv::imwrite("raw.png", frame);
		//return 0;
	}
	else
	{
		cap_usb >> frame;
		//cv::imwrite("raw.jpg", frame);
		//return 0;
	}
	Stopwatch T_v("0"), T_rec("1"), T_cap("2");
	T_v.Reset();	T_v.Start();
	T_rec.Reset();  T_rec.Start();
	T_cap.Reset();  T_cap.Start();
	unsigned int offset = 0;
	cv::Size show_sz(1500, 1000);
	cv::Size crop_sz(1300, 1000);
	cv::Rect roi((show_sz.width - crop_sz.width) / 2.0, (show_sz.height - crop_sz.height)/ 2.0, crop_sz.width, crop_sz.height);
	std::string fps;
	while (!frame.empty())
	{
		cv::resize(frame, frame, show_sz);
		frame = frame(roi);
		ice::KernelAlg::Instance()->OnUpdate(frame, offset);
		
		if(++offset % 10 == 0)
		{	
			T_v.Stop();
			float time = T_v.GetTime();	
			fps = std::string("fps:") + std::to_string(10 / time);
			LOG(INFO) << fps;
			T_v.Reset();	T_v.Start();
		}
		if(ice::g_upload_interval > 0)//record
		{
			T_rec.Stop();
			float itv = T_rec.GetTime();
			if(std::abs(itv - ice::g_upload_interval) < 0.1)
			{
				T_rec.Reset();  T_rec.Start();
				std::string shotname = "./Record/" + ice::getSystemTime() + "-StoreId-" + std::to_string(ice::g_store_id) + "-CameraId-" + std::to_string(ice::g_camera_id) +".jpg";
				LOG(INFO) << "---shot---" << shotname;
				cv::imwrite(shotname.c_str(), frame);
			}
			else
			{
				T_rec.Start();
			}
		}

		if(ice::g_show_video)
		{
			cv::putText(frame, fps.c_str(), cv::Point(0, 140), 1, 1, CV_RGB(255, 255, 255), 2);
			cv::putText(frame, std::string("Buf:") + std::to_string(cap_ksj.BufSize()).c_str(), cv::Point(0, 100), 1, 1, CV_RGB(255, 255, 255), 2);
			cv::imshow("camera-show", frame);
			if(cv::waitKey(1)==' ')
                cv::waitKey(0);
        }
        T_cap.Reset();  T_cap.Start();
        if(ice::g_use_ksj)
        {
            cap_ksj >> frame;
        }
        else
        {
            cap_usb >> frame;
        }
        T_cap.Stop();
        LOG(INFO) << "VideoCapture cost: " << T_cap.GetTime() << "ms";
	}

    if(ice::g_use_ksj)
    {
        cap_ksj.Stop();
    }
	return 0;
}
