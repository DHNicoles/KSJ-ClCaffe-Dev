#include "utils/util.h"

namespace ice
{
	//////////////////////////////////////////////////////////////////////////
	//global var
	//////////////////////////////////////////////////////////////////////////

	int g_store_id = -1;
	int g_camera_id = -1;
	int g_upload_interval = -1;//[-1, >0]
	extern bool g_use_ksj = true;//[true, false]
	bool g_show_video = false;//[true, false]
	float g_detect_thresh = 0.35;

	cv::Size g_detect_win(600, 400);
	int g_margin = 30;

	int parseCfg(const char* cfg)
	{
		LOG(INFO) << "parsing from " << cfg;
		std::ifstream file(cfg);
		char buffer[296];
		while (!file.eof())
		{
			file.getline(buffer, 256);
			std::stringstream line(buffer);
			if (line.str().empty()) continue;
			std::string str;
			line >> str;

			std::vector<std::string> elem_v;
			split(str, elem_v, '=');
			CHECK_EQ(elem_v.size(), 2) << "cfg format error : " << str;
			LOG(INFO) << elem_v[0] << ":" << elem_v[1];
			if (elem_v[0] == "StoreId") g_store_id = atoi(elem_v[1].c_str());
			else if (elem_v[0] == "CameraID") g_camera_id = atoi(elem_v[1].c_str());
			else if (elem_v[0] == "DetectThresh") g_detect_thresh = atof(elem_v[1].c_str());
			else if (elem_v[0] == "UploadInterval") g_upload_interval = atoi(elem_v[1].c_str());
			else if (elem_v[0] == "UseKSJVision") g_use_ksj = (elem_v[1] == "true");
			else if (elem_v[0] == "ShowVideo") g_show_video = (elem_v[1] == "true");
			else if (elem_v[0] == "Margin") g_margin = atoi(elem_v[1].c_str());
		}
		return 0;
	}
	void split(std::string& str, std::vector<std::string>& stringArr, char c)
	{
		std::string elem;
		for (size_t i = 0; i <= str.size(); ++i)
		{
			if (i == str.size() && !elem.empty())
			{
				stringArr.push_back(elem);
			}
			else if (str[i] == c)
			{
				if (!elem.empty()) stringArr.push_back(elem);
				elem.clear();
			}
			else
			{
				elem += str[i];
			}
		}
	}
	std::string getSystemTime()
	{
		time_t now = time(0);
		tm *ltm = localtime(&now);
		char timestr[256];
		sprintf(timestr, "%d-%02d-%02d-%02d-%02d-%02d",
			ltm->tm_year + 1900,
			ltm->tm_mon + 1,
			ltm->tm_mday,
			ltm->tm_hour,
			ltm->tm_min,
			ltm->tm_sec);
		return std::string(timestr);
		/*printf("%d-%02d-%02d %02d:%02d:%02d",
			st.wYear,
			st.wMonth,
			st.wDay,
			st.wHour,
			st.wMinute,
			st.wSecond);*/
	}

	double distanceOfIOU(cv::Rect & r1, cv::Rect & r2)
	{
		cv::Rect iou = (r1 & r2);
		cv::Rect un = (r1 | r2);
		double t = (double)iou.area() / un.area();
		return t;
	}
    int L1distanceOfCentroid(cv::Rect & r1, cv::Rect & r2)
	{
		int cent_x_1 = r1.x + (r1.width >> 1);
		int cent_y_1 = r1.y + (r1.height >> 1);
		int cent_x_2 = r2.x + (r2.width >> 1);
		int cent_y_2 = r2.y + (r2.height >> 1);
        int dx = std::abs(cent_x_1 - cent_x_2);
        int dy = std::abs(cent_y_1 - cent_y_2);
		return std::sqrt(dx * dx + dy * dy);
	}
	int distanceOfCentroid(cv::Rect & r1, cv::Rect & r2)
	{
		int cent_x_1 = r1.x + (r1.width >> 1);
		int cent_y_1 = r1.y + (r1.height >> 1);
		int cent_x_2 = r2.x + (r2.width >> 1);
		int cent_y_2 = r2.y + (r2.height >> 1);
		return std::abs(cent_x_1 - cent_x_2) + std::abs(cent_y_1 - cent_y_2);
	}
	int distanceOfCentroid(cv::Rect & r1, cv::Point & pos)
	{
		int cent_x_1 = r1.x + (r1.width >> 1);
		int cent_y_1 = r1.y + (r1.height >> 1);
		return std::abs(cent_x_1 - pos.x) + std::abs(cent_y_1 - pos.y);
	}
	int distanceOfCentroid(cv::Point & pos1, cv::Point & pos2)
	{
		return std::abs(pos1.x - pos2.x) + std::abs(pos1.y - pos2.y);
	}

	cv::Rect scaleRect(cv::Rect& r, float factor)
	{
		float w = r.width*factor;
		float h = r.height*factor;
		cv::Rect ret(0, 0, w, h);
		ret.x = r.x + (r.width - w) / 2;
		ret.y = r.y + (r.height - h) / 2;
		return ret;
	}

	/*cv::Rect subRect(cv::Rect& r, cv::Mat& frame)
	{
	cv::Rect roi;
	roi.x = r.x + r.width / 3;
	roi.y = r.y + r.height;
	roi.width = r.width / 3;
	roi.height = r.height;
	cv::Rect f(0, 0, frame.cols, frame.rows);
	return roi&f;
	}*/
	std::vector<cv::Rect> subRect(cv::Rect& r, cv::Mat& frame)
	{
		cv::Rect roi1, roi2, roi3;
		roi1.x = r.x + r.width / 8;
		roi1.y = r.y + r.height;
		roi1.width = r.width / 4;
		roi1.height = r.height / 2;

		roi2.x = r.x + r.width / 8 * 3;
		roi2.y = r.y + r.height;
		roi2.width = r.width / 4;
		roi2.height = r.height / 2;

		roi3.x = r.x + r.width - r.width / 8 * 5;
		roi3.y = r.y + r.height;
		roi3.width = r.width / 4;
		roi3.height = r.height / 2;

		cv::Rect f(0, 0, frame.cols, frame.rows);
		return{ roi1&f, roi2&f, roi3&f };
	}
	cv::Rect clothRect(cv::Rect& r, cv::Mat& frame)
	{
		cv::Rect roi;
		roi.x = r.x + r.width / 8 * 3;
		roi.y = r.y + r.height;
		roi.width = r.width / 4;
		roi.height = r.height;
		cv::Rect f(0, 0, frame.cols, frame.rows);
		return roi;
	}
	//////////////////////////////////////////////////////////////////////////
	//
	//Use the o-th and 1-st channels  
	int channels[] = { 0, 1, 2 };
	/// Using 100 bins for hue RBG
	int r_bins = 255;
	int g_bins = 255;
	int b_bins = 255;
	int histSize[] = { r_bins, g_bins, b_bins };
	float r_ranges[] = { 0, 256 };
	float g_ranges[] = { 0, 256 };
	float b_ranges[] = { 0, 256 };
	const float* ranges[] = { r_ranges, g_ranges, b_ranges };
	///////
	//hog//
	///////
	//cv::HOGDescriptor *hog = new cv::HOGDescriptor(cvSize(32, 32), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);  //构造HOG，具体意思见参考文章1,2            

	//////////////////////////////////////////////////////////////////////////
	bool matchRoi(cv::Mat& roi1, cv::Mat& roi2, int thresh)
	{
		double base_half = calcRoiDist(roi1, roi2);
		LOG(INFO) << "MatchRoi = " << base_half;
		//cv::imshow("roi1", roi1);
		//cv::imshow("roi2", roi2);
		//cv::waitKey(0);
		return base_half < thresh;
	}
	//比较两数大小  
	template <typename _Tp>
	int mem_cmp(const void *a, const void *b)
	{
		//当_Tp为浮点型，可能由于精度，会影响排序  
		return (*((_Tp *)a) - *((_Tp *)b));
	}

	//求Mat元素中值  
	template <typename _Tp>
	_Tp medianElem(cv::Mat img)
	{
		_Tp *buf;
		size_t total = img.total();

		buf = new _Tp[total];

		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				buf[i*img.cols + j] = img.ptr<_Tp>(i)[j];
			}
		}

		qsort(buf, total, sizeof(_Tp), mem_cmp<_Tp>);

		return buf[total / 2];
	}
	double calcRoiDist(cv::Mat& roi1, cv::Mat& roi2)
	{
		if (roi1.channels() != 3 || roi2.channels() != 3)
		{
			LOG(ERROR) << "roi1.channels()=" << roi1.channels();
			LOG(ERROR) << "roi2.channels()=" << roi2.channels();
			return -1;
		}
		//////////////////////////////////////////////////////////////////////////
		/// Histograms  
		/*cv::MatND hist_base_1, hist_base_2;
		/// Calculate the histograms for the HSV images
		cv::calcHist(&roi1, 1, channels, cv::Mat(), hist_base_1, 2, histSize, ranges, true, false);
		cv::normalize(hist_base_1, hist_base_1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
		cv::calcHist(&roi2, 1, channels, cv::Mat(), hist_base_2, 2, histSize, ranges, true, false);
		cv::normalize(hist_base_2, hist_base_2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
		double base_half = cv::compareHist(hist_base_1, hist_base_2, 1);
		return base_half;*/
		//////////////////////////////////////////////////////////////////////////
		//RGB tl bl tr br
		//////////////////////////////////////////////////////////////////////////
		/*cv::Rect tl_1(0, 0, roi1.cols >> 1, roi1.rows >> 1);
		cv::Rect bl_1(0, roi1.rows >> 1, roi1.cols >> 1, roi1.rows >> 1);
		cv::Rect tr_1(roi1.cols >> 1, 0, roi1.cols >> 1, roi1.rows >> 1);
		cv::Rect br_1(roi1.cols >> 1, roi1.rows >> 1, roi1.cols >> 1, roi1.rows >> 1);
		cv::Scalar s1_tl = cv::mean(roi1(tl_1));
		cv::Scalar s1_bl = cv::mean(roi1(bl_1));
		cv::Scalar s1_tr = cv::mean(roi1(tr_1));
		cv::Scalar s1_br = cv::mean(roi1(br_1));

		cv::Rect tl_2(0, 0, roi2.cols >> 1, roi2.rows >> 1);
		cv::Rect bl_2(0, roi2.rows >> 1, roi2.cols >> 1, roi2.rows >> 1);
		cv::Rect tr_2(roi2.cols >> 1, 0, roi2.cols >> 1, roi2.rows >> 1);
		cv::Rect br_2(roi2.cols >> 1, roi2.rows >> 1, roi2.cols >> 1, roi2.rows >> 1);
		cv::Scalar s2_tl = cv::mean(roi2(tl_2));
		cv::Scalar s2_bl = cv::mean(roi2(bl_2));
		cv::Scalar s2_tr = cv::mean(roi2(tr_2));
		cv::Scalar s2_br = cv::mean(roi2(br_2));

		double base_tl = abs(s1_tl.val[0] - s2_tl.val[0]) + abs(s1_tl.val[1] - s2_tl.val[1]) + abs(s1_tl.val[2] - s2_tl.val[2]);
		double base_bl = abs(s1_bl.val[0] - s2_bl.val[0]) + abs(s1_bl.val[1] - s2_bl.val[1]) + abs(s1_bl.val[2] - s2_bl.val[2]);
		double base_tr = abs(s1_tr.val[0] - s2_tr.val[0]) + abs(s1_tr.val[1] - s2_tr.val[1]) + abs(s1_tr.val[2] - s2_tr.val[2]);
		double base_br = abs(s1_br.val[0] - s2_br.val[0]) + abs(s1_br.val[1] - s2_br.val[1]) + abs(s1_br.val[2] - s2_br.val[2]);
		double base_half = base_tl + base_bl + base_tr + base_br;
		return base_half;*/
		//////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////
		//RGB meam
		cv::Scalar s1 = cv::mean(roi1);
		cv::Scalar s2 = cv::mean(roi2);
		double base_half = abs(s1.val[0] - s2.val[0]) + abs(s1.val[1] - s2.val[1]) + abs(s1.val[2] - s2.val[2]);
		return base_half;
		//////////////////////////////////////////////////////////////////////////
		//HOG
		/*cv::Mat src_1, src_2;
		resize(roi1, src_1, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC);
		resize(roi2, src_2, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC);

		std::vector<float> descriptors_1, descriptors_2;
		hog->compute(src_1, descriptors_1, cv::Size(1, 1), cv::Size(0, 0));
		hog->compute(src_2, descriptors_2, cv::Size(1, 1), cv::Size(0, 0));
		LOG(INFO) << "HOG dims: " << descriptors_1.size() << "," << descriptors_2.size();
		double base_half = 0.0;

		for (size_t i = 0; i < descriptors_1.size(); ++i)
		{
		base_half += abs(descriptors_1[i] - descriptors_2[i]);
		}*/

		//////////////////////////////////////////////////////////////////////////
		//h - median
		/*cv::Mat src_1, src_2;
		cv::cvtColor(roi1, src_1, CV_BGR2HSV);
		cv::cvtColor(roi2, src_2, CV_BGR2HSV);
		uchar mid_1 = medianElem<uchar>(src_1);
		uchar mid_2 = medianElem<uchar>(src_2);
		double base_half = abs(mid_1 - mid_2);*/

		//////////////////////////////////////////////////////////////////////////
		//

		//return base_half;
	}

	int getNewestERP(std::vector<std::string>& newErpVec)
	{
		return 0;
	}

	double calcTraitsDist(std::vector<cv::Point> points_1, std::vector<cv::Point> points_2)
	{
		//points_1 end N point
		//points_2 start N point
		//mid_point is mid between end and start
		cv::Point mid_point;
		mid_point.x = (points_1.back().x + points_2.front().x) >> 1;
		mid_point.y = (points_1.back().y + points_2.front().y) >> 1;

		cv::Vec4f line_para_1;
		cv::fitLine(points_1, line_para_1, CV_DIST_L2, 0, 1e-2, 1e-2);

		cv::Vec4f line_para_2;
		cv::fitLine(points_2, line_para_2, CV_DIST_L2, 0, 1e-2, 1e-2);

		//cacl dist between mid_point and line 1,2 
		cv::Point pointb_1;
		pointb_1.x = line_para_1[2];
		pointb_1.y = line_para_1[3];
		double cos_theta_1 = line_para_1[0];
		double sin_theta_1 = line_para_1[1];
		double x_1 = std::abs(pointb_1.x - mid_point.x);
		double y_1 = std::abs(pointb_1.y - mid_point.y);
		double mo_1 = std::sqrt(x_1*x_1 + y_1*y_1);
		double dist_1 = abs(x_1*cos_theta_1 + y_1*sin_theta_1) / mo_1;
		dist_1 *= std::sqrt(1 - dist_1*dist_1) * mo_1;

		cv::Point pointb_2;
		pointb_2.x = line_para_2[2];
		pointb_2.y = line_para_2[3];
		double cos_theta_2 = line_para_2[0];
		double sin_theta_2 = line_para_2[1];
		double x_2 = std::abs(pointb_2.x - mid_point.x);
		double y_2 = std::abs(pointb_2.y - mid_point.y);
		double mo_2 = std::sqrt(x_2*x_2 + y_2*y_2);
		double dist_2 = abs(x_2*cos_theta_2 + y_2*sin_theta_2) / mo_2;
		dist_2 *= std::sqrt(1 - dist_2*dist_2)* mo_2;

		double tend_x_1 = points_1.back().x - points_1.front().x;
		double tend_y_1 = points_1.back().y - points_1.front().y;
		double tend_mo_1 = std::sqrt(tend_x_1*tend_x_1 + tend_y_1*tend_y_1);
		double tend_x_2 = points_2.back().x - points_2.front().x;
		double tend_y_2 = points_2.back().y - points_2.front().y;
		double tend_mo_2 = std::sqrt(tend_x_2*tend_x_2 + tend_y_2*tend_y_2);
		double direct_dist_1 = 1 - (tend_x_1*tend_x_2 + tend_y_1*tend_y_2) / (tend_mo_1*tend_mo_2);

		double tend_x_3 = points_2.back().x - points_1.front().x;
		double tend_y_3 = points_2.back().y - points_1.front().y;
		double tend_mo_3 = std::sqrt(tend_x_3*tend_x_3 + tend_y_3*tend_y_3);
		double direct_dist_2 = 1 - (tend_x_1*tend_x_3 + tend_y_1*tend_y_3) / (tend_mo_1*tend_mo_3);
		return (dist_1 + dist_2) / 10 + (direct_dist_1 + direct_dist_2) * 10;

	}
	// Serialize a cv::Mat to a stringstream
	std::stringstream serialize(cv::Mat input)
	{
		// We will need to also serialize the width, height, type and size of the matrix
		int width = input.cols;
		int height = input.rows;
		int type = input.type();
		size_t size = input.total() * input.elemSize();

		// Initialize a stringstream and write the data
		std::stringstream ss;
		ss.write((char*)(&width), sizeof(int));
		ss.write((char*)(&height), sizeof(int));
		ss.write((char*)(&type), sizeof(int));
		ss.write((char*)(&size), sizeof(size_t));

		// Write the whole image data
		ss.write((char*)input.data, size);

		return ss;
	}
	// Deserialize a Mat from a stringstream
	cv::Mat deserialize(std::stringstream& input)
	{
		// The data we need to deserialize
		int width = 0;
		int height = 0;
		int type = 0;
		size_t size = 0;

		// Read the width, height, type and size of the buffer
		input.read((char*)(&width), sizeof(int));
		input.read((char*)(&height), sizeof(int));
		input.read((char*)(&type), sizeof(int));
		input.read((char*)(&size), sizeof(size_t));

		// Allocate a buffer for the pixels
		char* data = new char[size];
		// Read the pixels from the stringstream
		input.read(data, size);

		// Construct the image (clone it so that it won't need our buffer anymore)
		cv::Mat m = cv::Mat(height, width, type, data).clone();

		// Delete our buffer
		delete[]data;

		// Return the matrix
		return m;
	}

	void saveImageByIndex(cv::Mat& roi, size_t index, std::string& info)
	{
		std::string dir = "../../resource/FrontFace/";
		dir += std::to_string(index);
		static std::map<size_t, size_t> cnt_mp;
		int status = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		//if(access(dir.c_str, NULL)!=0)  
		if (status == 0)
		{
			//dir += "/" + info + "_" + std::to_string(++cnt_mp[index]) + ".jpg";
			dir += "/" + info + "_" + getSystemTime() + ".jpg";
			LOG(INFO) << "write image name " << dir;
			cv::imwrite(dir.c_str(), roi);
		}
		else
		{
			LOG(ERROR) << "mkdir " << dir << " error";
		}
	}
}

