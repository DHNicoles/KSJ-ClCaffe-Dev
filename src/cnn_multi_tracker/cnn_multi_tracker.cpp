#include "cnn_multi_tracker.h"
#include "../game/object_info.h"
#include "detector/detector.h"
namespace ice
{
	CNNMultiTracker::CNNMultiTracker() :
		key_max_(0)
	{
		OnInit();
	}
	CNNMultiTracker::~CNNMultiTracker()
	{
		OnDestroy();
	}
	void CNNMultiTracker::OnInit()
	{

	}
	void CNNMultiTracker::OnDestroy()
	{
		//delete object
		std::map<size_t, ObjectInfo*>::iterator obj_itr = object_info_map_.begin();
		std::map<size_t, ObjectInfo*>::iterator obj_end_itr = object_info_map_.end();
		for (; obj_itr != obj_end_itr; ++obj_itr)
		{
			if (obj_itr->second)
				delete obj_itr->second;
			obj_itr->second = NULL;
		}
	}
	bool CNNMultiTracker::IsOutboard(cv::Rect & positionBox)
    {
        int cent_x = positionBox.x + (positionBox.width >> 1);
		int cent_y = positionBox.y + (positionBox.height >> 1);
        //check if out of board
        if (
			cent_x < 0 || // outboard
			cent_y < 0 ||
			cent_x >= bound_mask_.cols ||
			cent_y >= bound_mask_.rows ||
			bound_mask_.at<uchar>(cent_y, cent_x) == 0 
		   )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
	void CNNMultiTracker::AddTracker(cv::Mat & frame, cv::Rect & positionBox)
	{
		positionBox &= cv::Rect(0, 0, frame.cols, frame.rows);
		int cent_x = positionBox.x + (positionBox.width >> 1);
		int cent_y = positionBox.y + (positionBox.height >> 1);
        //check if out of board
        if (IsOutboard(positionBox)) return;
        //else
		int index = GetIndex_();
		LOG(INFO) << "add tracker GetIndex_=" << index <<", loc: " << positionBox;
		object_info_map_[index] = new ObjectInfo(index, cv::Point(cent_x, cent_y), positionBox);
    }
    void CNNMultiTracker::OnUpdate(cv::Mat & frame)
    {
        cv::Mat highest_mat = frame.clone();
        static Stopwatch T_detect("detect");
        LOG(INFO) << "Trackers number: " << object_info_map_.size();
        //net forward to get every object newest loc.
        static std::vector<std::vector<cv::Rect> > obj_boxes;
        static std::vector<cv::Rect> scale_rect_bais;
        static std::vector<std::vector<float> > score;
        static std::vector<cv::Mat> mat_vec;
        mat_vec.clear(), obj_boxes.clear(),	score.clear(), scale_rect_bais.clear();
        //////////////////////////////////////////////////////////////////
		//For 1: clear objects when Iou > thresh over objs themselves; 
        //For 2: check the probility of collision between objects;
        //
        //Reset SearchRatio
        for(auto& iter : object_info_map_) iter.second->SearchRatio() = 2.0;
        std::map<size_t, ObjectInfo*>::iterator obj_itr = object_info_map_.begin();
        std::map<size_t, ObjectInfo*>::iterator obj_end_itr = object_info_map_.end();
		for (; obj_itr != obj_end_itr; ++obj_itr)
		{
			if(obj_itr->second->IsShit() == true) continue;
			std::map<size_t, ObjectInfo*>::iterator obj_j = obj_itr;
			++obj_j;
			for (; obj_j != obj_end_itr; ++obj_j)
			{
                //For 1
				cv::Rect in = (obj_itr->second->Box() & obj_j->second->Box());
				float t = max((float)in.area() / obj_itr->second->Box().area(), (float)in.area() / obj_j->second->Box().area());
				if (t > 0.8)
				{
					//the later one  of them should be shit
					//obj_itr->second->IsShit() = true;
					obj_j->second->IsShit() = true;
				}

                //For 2 
                static double factor = 1.5;
                int dist = L1distanceOfCentroid(obj_itr->second->Box(), obj_j->second->Box());
                double ratio_w_u = (float)dist / obj_itr->second->Box().width * factor; 
                double ratio_h_u = (float)dist / obj_itr->second->Box().height * factor;
                double ratio_u = min(ratio_w_u, ratio_h_u);
                obj_itr->second->SearchRatio() = min(obj_itr->second->SearchRatio(), ratio_u);
                double ratio_w_v = (float)dist / obj_j->second->Box().width * factor;
                double ratio_h_v = (float)dist / obj_j->second->Box().height * factor;
                double ratio_v = min(ratio_w_v, ratio_h_v);
                obj_j->second->SearchRatio() = min(obj_j->second->SearchRatio(), ratio_v);
                //LOG(INFO) << "[dist, obj_itr->second->Box().width, height]: " << dist << ", " << obj_itr->second->Box().width << "," << obj_itr->second->Box().height;
                //LOG(INFO) << "[ratio_w, ratio_h, obj_itr->second->SearchRatio()]: " << ratio_w << ", " << ratio_h << "," <<obj_itr->second->SearchRatio();
            }
            //For 2 , at least expand to 1.2, and at most expand to 2.0
            obj_itr->second->SearchRatio() = max(obj_itr->second->SearchRatio(), 1.4);
            obj_itr->second->SearchRatio() = min(obj_itr->second->SearchRatio(), 2.0);

		}
        //////////////////////////////////////////////////////////////////
        //update cnn tracker
		obj_itr = object_info_map_.begin();
        for (; obj_itr != obj_end_itr; ++obj_itr)
        {
            cv::Rect& loc = obj_itr->second->Box();    
            cv::Rect scale_rect = scaleRect(loc, 2.0);
            cv::Rect actual_rect = scaleRect(loc, obj_itr->second->SearchRatio());
            scale_rect &= cv::Rect(0, 0, frame.cols, frame.rows);
            actual_rect &= cv::Rect(0, 0, frame.cols, frame.rows);
            scale_rect_bais.push_back(scale_rect);
            cv::Mat actual_roi = frame(actual_rect);
            cv::Mat merged(cv::Size(scale_rect.width, scale_rect.height), actual_roi.type());
            merged = cv::Scalar::all(127.5);
            //cv::imshow("a-merged", merged);
            cv::Rect actual_loc((scale_rect.width - actual_rect.width) / 2.0, (scale_rect.height - actual_rect.height) / 2.0, actual_rect.width, actual_rect.height); 
            actual_roi.copyTo(merged(actual_loc));
            mat_vec.push_back(merged);
            cv::imshow("b-merged", merged);
            //cv::Mat scale_roi = frame(scale_rect);
            //cv::imshow("roix2", scale_roi);
            //cv::waitKey(0);
            //cv::waitKey(0);
        }
        T_detect.Reset();   T_detect.Start();
        Detector::Instance()->Detect(mat_vec, obj_boxes, score, Detector::DETECTOR_SCALE_LOCAL);
        T_detect.Stop();
        LOG(INFO) << "local model detection cost: " << T_detect.GetTime();
        obj_itr = object_info_map_.begin();
        for (int i = 0; obj_itr != obj_end_itr; ++obj_itr, ++i)
        {
            //LOG(INFO) << "scale_roi detect number: " << obj_boxes.size(); 

            if(!obj_boxes[i].empty())
            {
                //std::cout << "score:" <<std::endl;
                //for(auto& s:score[i]) std::cout << s << ",";
                //std::cout << "box:" <<std::endl;
                //for(auto& r:obj_boxes[i]) std::cout << r << ",";
                //std::cout << std::endl;
                //find top 1 score and box, update object loc.
                std::vector<float>::iterator biggest = std::max_element(std::begin(score[i]), std::end(score[i])); 
                cv::Rect& highest_rect = obj_boxes[i].at(std::distance(std::begin(score[i]), biggest));
                //LOG(INFO) << "-scale_rect box: " << scale_rect_bais[i];
                //LOG(INFO) << "-highest rect: " << highest_rect;
                //highest_rect &= cv::Rect(0, 0, scale_rect.width, scale_rect.height);
                //if(highest_rect.width == 0 || highest_rect.height == 0) continue;
                //cv::rectangle(mat_vec[i], highest_rect, cv::Scalar(255, 0, 0), 2);
                //cv::imshow("scale", mat_vec[i]);
                highest_rect.x += scale_rect_bais[i].x;
                highest_rect.y += scale_rect_bais[i].y;
                int cent_x = highest_rect.x + (highest_rect.width >> 1);
                int cent_y = highest_rect.y + (highest_rect.height >> 1);
                obj_itr->second->GetTrace().push_back(cv::Point(cent_x, cent_y));
                obj_itr->second->Box() = highest_rect;
                obj_itr->second->ClearMiss();
                //cv::Mat img = frame.clone();
                cv::rectangle(highest_mat, highest_rect, cv::Scalar(255, 250, 250), 2);
                //cv::imshow("highest", img);
                //cv::waitKey(0);

            }
            else
            {
                //could not find any detection around it, maybe the obj is background.
                //Here should be marked.
                obj_itr->second->AddToMiss();
            }
            //cv::waitKey(0);
            obj_itr->second->OnUpdate(bound_mask_, frame);

		}
        cv::imshow("highest", highest_mat);
        
    }


	void CNNMultiTracker::Mark(size_t index)
	{

	}
	void CNNMultiTracker::RemoveInvalid()
	{
		//search invalid object's index
		std::map<size_t, ObjectInfo*>::iterator obj_itr = object_info_map_.begin();
		std::map<size_t, ObjectInfo*>::iterator obj_end_itr = object_info_map_.end();
		for (; obj_itr != obj_end_itr; ++obj_itr)
		{
			if (obj_itr->second->IsShit())
				invalid_index_list_.push_back(obj_itr->first);
		}
		//move invalid object to recycle by index
		std::list<size_t>::iterator iil_it = invalid_index_list_.begin();
		std::list<size_t>::iterator iil_end_it = invalid_index_list_.end();
		for (; iil_it != iil_end_it; ++iil_it)
		{
			size_t idx = *iil_it;
			//LOG(INFO) << "remove tracker index : " << idx;
			RemoveTrackerByIndex_(idx);
			//index_pool_set_.insert(idx);
		}
		invalid_index_list_.clear();
	}
	void CNNMultiTracker::RemoveTrackerByIndex_(size_t index)
	{
		std::map<size_t, ObjectInfo*>::iterator obj_itr = object_info_map_.find(index);
		if (obj_itr != object_info_map_.end())
		{
			//obj_itr->second->SetEndTime(getSystemTime());
			delete obj_itr->second;
			object_info_map_.erase(obj_itr);
		}
	}
	void CNNMultiTracker::Replace(cv::Mat & frame, cv::Rect & positionBox, size_t index)
	{

	}
	std::vector<cv::Scalar> color_pool =
	{
		CV_RGB(255, 0, 0),
		CV_RGB(255, 128, 0),
		CV_RGB(0, 255, 0),
		CV_RGB(0, 255, 255),
		CV_RGB(128, 0, 255),
	};
	void CNNMultiTracker::DrawTrace(cv::Mat& frame)
	{
		std::map<size_t, ObjectInfo*>::iterator obj_itr = object_info_map_.begin();
		std::map<size_t, ObjectInfo*>::iterator obj_end_itr = object_info_map_.end();
		for (; obj_itr != obj_end_itr; ++obj_itr)
		{
			//if (obj_itr->second->CanBeRecycle() == true)
			{
				std::list<cv::Point>::iterator point_it = obj_itr->second->GetTrace().begin();
				if (obj_itr->second->GetTrace().size() > 20)
				{
					std::advance(point_it, obj_itr->second->GetTrace().size() - 20);
				}
				std::list<cv::Point>::iterator point_end_it = obj_itr->second->GetTrace().end();
				for (; point_it != point_end_it; ++point_it)
				{
					cv::circle(frame, *point_it, 0, color_pool[obj_itr->first % color_pool.size()], 2, 1);
				}
				cv::Rect banner(obj_itr->second->Box().x - 1, obj_itr->second->Box().y - 20, obj_itr->second->Box().width + 2, 20);
				banner &= cv::Rect(0, 0, frame.cols, frame.rows);
				frame(banner) = color_pool[obj_itr->first % color_pool.size()];
				cv::rectangle(frame, obj_itr->second->Box(), color_pool[obj_itr->first % color_pool.size()], 2, 1, 0);
				//cv::putText(frame, std::to_string(obj_itr->first).c_str(), cv::Point(obj_itr->second->Box().x, obj_itr->second->Box().y - 4), 1, 0.9, CV_RGB(0, 0, 0), 1);
				//cv::putText(frame, std::to_string(obj_itr->second->SynceOver()).c_str(), cv::Point(obj_itr->second->Box().x + 10, obj_itr->second->Box().y + 20), 2, 0.5, color_pool[obj_itr->first % color_pool.size()], 1);
				cv::putText(frame, std::to_string(obj_itr->second->Life()).c_str(), cv::Point(obj_itr->second->Box().x + 10, obj_itr->second->Box().y + 30), 2, 0.4, color_pool[obj_itr->first % color_pool.size()], 1);
			}
		}
	}

	////////////////////////////////////////////////////////////
	//Draw UI in a high-quality image
	////////////////////////////////////////////////////////////
	void CNNMultiTracker::DrawUI(cv::Mat& frame)
	{
        int mg_x = g_margin * ((float)frame.cols / g_detect_win.width);
        int mg_y = g_margin * ((float)frame.rows / g_detect_win.height) * 0.7;
		cv::Rect bound(mg_x, mg_y,
			frame.cols - 2 * mg_x, frame.rows - 2 * mg_y);
		cv::rectangle(frame, bound, CV_RGB(125,255, 190), 3, 1, 0);
		std::map<size_t, ObjectInfo*>::iterator obj_itr = object_info_map_.begin();
		std::map<size_t, ObjectInfo*>::iterator obj_end_itr = object_info_map_.end();
		for (; obj_itr != obj_end_itr; ++obj_itr)
		{
			if (obj_itr->second->IsShit() == false)
			{
				std::list<cv::Point>::iterator point_it = obj_itr->second->GetTrace().begin();
				if (obj_itr->second->GetTrace().size() > 20)
				{
					std::advance(point_it, obj_itr->second->GetTrace().size() - 20);
				}
				std::list<cv::Point>::iterator point_end_it = obj_itr->second->GetTrace().end();
				static float fractor_x = (float)frame.cols / g_detect_win.width; 
				static float fractor_y = (float)frame.rows / g_detect_win.height;
				for (; point_it != point_end_it; ++point_it)
				{
					cv::Point dot(int(point_it->x * fractor_x), int(point_it->y * fractor_y));
					cv::circle(frame, dot, 0, color_pool[obj_itr->first % color_pool.size()], 2, 1);
				}
				cv::Rect head_box;
				head_box.x = int(obj_itr->second->Box().x * fractor_x);
				head_box.y = int(obj_itr->second->Box().y * fractor_y);
				head_box.width = int(obj_itr->second->Box().width * fractor_x);
				head_box.height = int(obj_itr->second->Box().height * fractor_y);
				cv::Rect banner(head_box.x - 1, head_box.y - 20, head_box.width + 2, 20);
				banner &= cv::Rect(0, 0, frame.cols, frame.rows);
				frame(banner) = color_pool[obj_itr->first % color_pool.size()];
				cv::rectangle(frame, head_box, color_pool[obj_itr->first % color_pool.size()], 2, 1, 0);
				cv::putText(frame, std::string("ID ") + std::to_string(obj_itr->first).c_str(), cv::Point(head_box.x, head_box.y - 4), 1, 0.9, CV_RGB(0, 0, 0), 1);
                cv::putText(frame, std::string("Ratio  ") + std::to_string(obj_itr->second->SearchRatio()).substr(0, 4).c_str(), cv::Point(head_box.x, head_box.y + 15), 2, 0.4, CV_RGB(255, 2, 215), 1);
				cv::putText(frame, std::string("MissTime ") + std::to_string(obj_itr->second->MissTime()).c_str(), cv::Point(head_box.x, head_box.y + 35), 2, 0.4, color_pool[obj_itr->first % color_pool.size()], 1);
			}
		}
	}

	void CNNMultiTracker::SetAllMatchFalse()
	{
		std::map<size_t, ObjectInfo*>::iterator obj_itr = object_info_map_.begin();
		std::map<size_t, ObjectInfo*>::iterator obj_end_itr = object_info_map_.end();
		for (; obj_itr != obj_end_itr; ++obj_itr)
		{
			obj_itr->second->SetMatchFalse();
		}
	}
	void CNNMultiTracker::SetMatchTrue(size_t index)
	{
		std::map<size_t, ObjectInfo*>::iterator obj_itr = object_info_map_.find(index);
		//if(obj_itr != object_info_map_.end())
			obj_itr->second->SetMatchTrue();
	}
	void CNNMultiTracker::SetBound(cv::Size size, cv::Rect bound)
	{
		bound_mask_ = cv::Mat(size, CV_8UC1);
		bound_mask_ = cv::Scalar::all(0);
		bound_mask_(bound) = cv::Scalar::all(255);
        //cv::imshow("bound", bound_mask_);
        //cv::waitKey(0);
	}


	int CNNMultiTracker::GetIndex_()
	{

		if (index_pool_set_.empty())
		{
			return ++key_max_;
		}
		else
		{
			int index = *(index_pool_set_.begin());
			index_pool_set_.erase(index_pool_set_.begin());
			return index;
		}
	}
}
