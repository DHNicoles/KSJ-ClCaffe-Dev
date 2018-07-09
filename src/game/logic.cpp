#include "game/logic.h"
#include "game/object_info.h"
#include "detector/detector.h"
namespace ice
{
	////////////////////////////////////
	//local def
	////////////////////////////////////
	double l_iou_thresh = 0.1;
	unsigned char l_global_detect_interval = 3;

	KernelAlg::KernelAlg() :
		detector_ptr_(NULL), Inited(false)
	{
		OnInit();
		cv::Rect bound(g_margin, g_margin * 0.7,
			g_detect_win.width - 2 * g_margin, g_detect_win.height - 2 * g_margin * 0.7);
		//set detector_win, trackor_win
		SetBound(g_detect_win, bound);
	}

	KernelAlg::~KernelAlg()
	{
		OnDestroy();
		Inited = false;
	}

	void KernelAlg::OnInit()
	{
		detector_ptr_ = new Detector;
		detector_ptr_->OnInit();


		cnn_multi_trackers_ptr_ = new CNNMultiTracker;
		cnn_multi_trackers_ptr_->OnInit();

		Inited = true;
	}
	void KernelAlg::OnDestroy()
	{
		if (detector_ptr_)
		{
			delete detector_ptr_;
		}
		detector_ptr_ = NULL;

		if (cnn_multi_trackers_ptr_)
		{
			delete cnn_multi_trackers_ptr_;
		}
		cnn_multi_trackers_ptr_ = NULL;

		//////////////////////////////////////////////////////////////////////////
	}

	void KernelAlg::OnUpdate(cv::Mat& frame, size_t offset)
	{
		static unsigned char t_g = 0;
		static unsigned char t_v = 0;
		static std::vector<std::vector<cv::Rect> > obj_boxes;
		static std::vector<std::vector<float> > score;
        static std::vector<cv::Mat> mat_vec;
		obj_boxes.clear(),	score.clear(), mat_vec.clear();
		cv::Mat crop_frame;
		cv::resize(frame, crop_frame, g_detect_win);
        mat_vec.push_back(crop_frame);
		//Timer
		static Stopwatch timer;
		timer.Reset();	timer.Start();
		/////////////////////////////////////////////////////////
		//trackers updating
		CNNMultiTracker::Instance()->OnUpdate(crop_frame);
		CNNMultiTracker::Instance()->RemoveInvalid();

		timer.Stop();
		LOG(INFO) << "update trackers\t=\t" << timer.GetTime();
        
        ////////////////////////////////////////////////////////
        //Chegf Big model detector
		if((t_g % l_global_detect_interval) == 0)
		{
			/////////////////////////////////////////////////////////
			//detector updating
			timer.Reset();	timer.Start();
			//Note : input size = l_windows size
			//Detector::Instance()->Detect(crop_frame, obj_boxes, score, Detector::DETECTOR_SCALE_GLOBAL);
            Detector::Instance()->Detect(mat_vec, obj_boxes, score, Detector::DETECTOR_SCALE_GLOBAL);
            CHECK_EQ(obj_boxes.size(), 1) << "Big model detect output none."; 
			timer.Stop();
			//LOG(INFO) << "detector num\t=\t" << obj_boxes.size();
			LOG(INFO) << "global model detector cost\t=\t" << timer.GetTime();

			//////////////////////////////////////////////////////////////////////////
			//matcher
			Match(obj_boxes[0], crop_frame);
			t_g = 1;
            //DRAW_DETECTOR
            if (true)
            {
                for (size_t i = 0; i < obj_boxes[0].size(); ++i)
                {
                    cv::rectangle(crop_frame, obj_boxes[0][i], cv::Scalar(255, 255, 255), 2, 1, 0);
                    std::string scoreStr = std::to_string(score[0][i]);
                    cv::putText(crop_frame, scoreStr.c_str(), cv::Point(obj_boxes[0][i].x + (obj_boxes[0][i].width >> 2), (obj_boxes[0][i].y + (obj_boxes[0][i].height >> 2))), 1, 1, CV_RGB(255, 255, 0), 2);
                }
                if(!obj_boxes[0].empty()) cv::imshow("Debug-Detector", crop_frame);
            }
        }
        else
        {
			//LOG(INFO) << "interval_t\t=\t" << (int)t;
			t_g += 1;
		}

        		
		//////////////////////////////////////////////////////////////////////////
		
		//DRAW_TRACKER
		if (true)
		{
			CNNMultiTracker::Instance()->DrawUI(frame);
			std::string time_str = getSystemTime();
			cv::putText(frame, time_str.c_str(), cv::Point(0, 14), 1, 1, CV_RGB(25, 255, 215), 1);
		}

	}
	void KernelAlg::SetBound(cv::Size size, cv::Rect r)
	{
		CNNMultiTracker::Instance()->SetBound(size, r);
	}


	void KernelAlg::Match(std::vector<cv::Rect>& detect_boxes, cv::Mat& frame)
	{

		CNNMultiTracker::Instance()->SetAllMatchFalse();

		std::map<size_t, ObjectInfo*>& tracking_obector_map = CNNMultiTracker::Instance()->TrackingObectorMap();

		for (size_t i = 0; i < detect_boxes.size(); ++i)
		{
			std::map<size_t, ObjectInfo*>::iterator tom_it = tracking_obector_map.begin();
			std::map<size_t, ObjectInfo*>::iterator tom_end_it = tracking_obector_map.end();
			std::map<size_t, ObjectInfo*>::iterator closest_it = tom_end_it;
			int min_dist = 1000000;

			for (; tom_it != tom_end_it; ++tom_it)
			{
				int dist = distanceOfCentroid(detect_boxes[i], tom_it->second->Position());
				if (dist < min_dist && dist < DIST_HUMAN)
				{
					min_dist = dist;
					closest_it = tom_it;
				}
			}
			if (closest_it == tom_end_it)
			{
				LOG(INFO) << "add to tracker";
				CNNMultiTracker::Instance()->AddTracker(frame, detect_boxes[i]);
			}
			else
			{
				CNNMultiTracker::Instance()->SetMatchTrue(closest_it->first);
#ifdef RECTIFY
				cv::Rect iou = (closest_it->second->Box() & detect_boxes[i]);
				cv::Rect un = (closest_it->second->Box() | detect_boxes[i]);
				double t = (double)iou.area() / un.area();
				if (t < 0.9)
				{
					//LOG(INFO) << "replace tracker.index = " << closest_it->first;
					CNNMultiTracker::Instance()->Replace(frame, detect_boxes[i], closest_it->first);
				}
#endif
			}
		}
	}
	void KernelAlg::Match_ex(std::vector<cv::Rect>& detect_boxes, cv::Mat& frame)
	{
		//Global detector fingers out all heads in whole scope,
		//macro RECTIFY control:
		//if we trust local detector, only add operator will be performed.
		//if we trust global one, both add and rectify operator will be performed.

		CNNMultiTracker::Instance()->SetAllMatchFalse();

		std::map<size_t, ObjectInfo*>& tracking_objector_map = CNNMultiTracker::Instance()->TrackingObectorMap();

		for (int i = 0; i < detect_boxes.size(); ++i)
		{
			std::map<size_t, ObjectInfo*>::iterator tom_it = tracking_objector_map.begin();
			std::map<size_t, ObjectInfo*>::iterator tom_end_it = tracking_objector_map.end();
			std::map<size_t, ObjectInfo*>::iterator closest_iter = tracking_objector_map.end();
			double max_iou = l_iou_thresh;//0.3
			//searching for the most closest one in tracker boxes
			for (; tom_it != tom_end_it; ++tom_it)
			{
				double iou = distanceOfIOU(detect_boxes[i], tom_it->second->Box());
				if (iou > max_iou)
				{
					max_iou = iou;
					closest_iter = tom_it;
				}
			}

			//if there's no one was matched, add this to traker.
			if (closest_iter == tom_end_it)
			{
				LOG(INFO) << "add to tracker";
				CNNMultiTracker::Instance()->AddTracker(frame, detect_boxes[i]);
			}
			//else, check if the closest box is more suitable
			else
			{
				CNNMultiTracker::Instance()->SetMatchTrue(closest_iter->first);
#ifdef RECTIFY
				if (max_iou < 0.9)
				{
					//LOG(INFO) << "replace tracker.id = " << closest_iter->first;
					CNNMultiTracker::Instance()->Replace(frame, detect_boxes[i], closest_iter->first);
				}
#endif
			}
		}
	}
}
