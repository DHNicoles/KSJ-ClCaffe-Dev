#include "game/object_info.h"

namespace ice
{
	/////////
	//local v
	////////
	cv::RNG l_rng(0xFFFF);

	pthread_mutex_t l_mutex_getback_lock = PTHREAD_MUTEX_INITIALIZER;

	//////////////////////////

	ObjectInfo::ObjectInfo(int id, cv::Point point, cv::Rect& box) :
		trace_(1, point), newest_box_(box)
	{
		//time_vec[0] = getSystemTime();
		index_ = id;
		match_success_ = true;
		true_shit_ = false;
		life_ = l_life_begin;
        missed_ = 0;
        search_ratio_ = 2.0; 
		match_time_ = 0;
		decay_ = 1;
		max_score_ = 0;
		max_displace_ = 0;
		
	}
	void ObjectInfo::OnUpdate(cv::Mat& bound_mask, cv::Mat & frame)
	{
		if (this->true_shit_) return;
		cv::Point& center = Position();
		/////////////////////////////
		///update max displace
		cv::Point& begin = trace_.front();
		int d = abs(center.x - begin.x) + abs(center.y - begin.y); 
		max_displace_ = max_displace_ > d ? max_displace_ : d;
		if(newest_box_.area() > 15000) true_shit_ = true;
		/////////////////////////////
		if (
			center.x < 0 || // outboard
			center.y < 0 ||
			center.x >= bound_mask.cols ||
			center.y >= bound_mask.rows ||
			bound_mask.at<uchar>(center) == 0 ||
            missed_ >= 10 //small model detect nothing
		   )
		{
			true_shit_ = true;
		}
		else
		{
		}
	}
	std::ostream& operator << (std::ostream& os, ObjectInfo & obj)
	{
		//os << "OBJ INFO: \nindex =\t\t" << obj.index_ << "\nstart time =\t" << obj.time_vec[0] << std::endl;
		//os << "end time =\t" << obj.time_vec[1] << std::endl;
		os << "Trace length =\t" << obj.GetTrace().size() << std::endl;
		os << "max_score =\t" << obj.MaxScore() << std::endl;
		return os;
	}
	ObjectInfo::~ObjectInfo()
	{
		OnDestory();
	}
	void ObjectInfo::OnDestory()
	{
	}

}
