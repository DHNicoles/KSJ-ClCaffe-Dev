/************************************************************************/
/* tools                                                                */
/************************************************************************/

#ifndef __UTIL_H__
#define __UTIL_H__

#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>     
#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <utility>
#include <deque>
#include <chrono>
#include <future>
#include <glog/logging.h>
#include "Stopwatch.hpp"
#include "CBase64Coder.h"
#ifndef WIN32
#include "pthread.h"
#include <unistd.h> 
#include <sys/stat.h>
#endif
#ifdef WIN32
#include "stdio.h"
#include "conio.h"
#include <direct.h>
#include <io.h>
#endif // WIN32
/*
#ifndef max
#define max(a,b)	(((a) > (b)) ? (a) : (b))
#endif

#ifndef min 
#define min(a,b)	(((a) < (b)) ? (a) : (b))
#endif
 */
namespace ice
{
	//////////////////////////////////////////////////////////////////////////
	//global var
	//////////////////////////////////////////////////////////////////////////

	extern int g_store_id;
	extern int g_camera_id;

	extern float g_detect_thresh;//[0, 1]
	extern int g_upload_interval;//[-1, >0]
	extern bool g_use_ksj;//[true, false]
	extern bool g_show_video;//[true, false]
	extern cv::Size g_detect_win;
	extern int g_margin;

	/************************************************************************/
	/* 
	filepath	:	read file names from 
	videoList	:	save file name list
	 */
	/************************************************************************/
	int  getFileList(std::string filepath, std::vector<std::string> &videoList);
	/************************************************************************/
	/* return codes                                                         */
	/************************************************************************/
	enum RETCODES
	{
		RETCODES_SUCCESS = 0,
		RETCODES_NULLPTR,
		RETCODES_INIT_ERROR,
	};
	int parseCfg(const char* cfg);

	void split(std::string& str, std::vector<std::string>& stringArr, char c);
	std::string getSystemTime();
	double distanceOfIOU(cv::Rect & r1, cv::Rect & r2);
    int L1distanceOfCentroid(cv::Rect & r1, cv::Rect & r2);
	int distanceOfCentroid(cv::Rect& r1, cv::Rect& r2);
	int distanceOfCentroid(cv::Rect& r1, cv::Point& pos);
	int distanceOfCentroid(cv::Point& pos1, cv::Point& pos2);
	cv::Rect scaleRect(cv::Rect& r, float factor);
	//cv::Rect subRect(cv::Rect& r, cv::Mat& frame);
	std::vector<cv::Rect> subRect(cv::Rect& r, cv::Mat& frame);
	cv::Rect clothRect(cv::Rect& r, cv::Mat& frame);
	bool matchRoi(cv::Mat& roi1,cv::Mat& roi2, int thresh);
	double calcRoiDist(cv::Mat& roi1, cv::Mat& roi2);
	int getNewestERP(std::vector<std::string>& newErpVec);
	double calcTraitsDist(std::vector<cv::Point> points_1, std::vector<cv::Point> points_2);
	// Serialize a cv::Mat to a stringstream
	std::stringstream serialize(cv::Mat input);
	// Deserialize a Mat from a stringstream
	cv::Mat deserialize(std::stringstream& input);
	void saveImageByIndex(cv::Mat& roi, size_t index, std::string& info);
}

/************************************************************************/
/* macro used in this global scope                                      */
/************************************************************************/


//match distance
#define	DIST_HUMAN	70

//////////////////////////////////////////////////////////////////////////

#define INTERVAL 1 

#endif // __UTIL_H__

