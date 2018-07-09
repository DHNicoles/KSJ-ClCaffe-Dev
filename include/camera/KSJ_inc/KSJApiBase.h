#ifndef __KSJ_API_BASE_H__
#define __KSJ_API_BASE_H__

#if defined (_WIN32)
#  ifdef KSJAPI_EXPORTS
#    define KSJAPI_IMPORT_EXPORT __declspec(dllexport)
#  else
#    define KSJAPI_IMPORT_EXPORT __declspec(dllimport)
#  endif

#  if defined (_M_IX86) || defined (__i386__)
#    define KSJAPI_CALLTYPE __stdcall
#  else
#    define KSJAPI_CALLTYPE  
#  endif

#elif defined (__GNUC__) && (__GNUC__ >= 4) && (defined (__linux__) || defined (__APPLE__))
#  define KSJAPI_IMPORT_EXPORT __attribute__((visibility("default")))
#  if defined (__i386__)
#    define KSJAPI_CALLTYPE __attribute__((stdcall))
#  else
#    define KSJAPI_CALLTYPE /* default */
#  endif
#  ifndef EXTERN_C
#    define EXTERN_C extern "C"
#  endif

#else
#  error Unknown platform, file needs adaption
#endif


#ifndef _WIN32
#define KSJ_API int
#else
#define KSJ_API KSJAPI_IMPORT_EXPORT int KSJAPI_CALLTYPE
#endif

#ifdef __cplusplus
extern "C"{
#endif
#define  TCHAR char
#define  HWND   void*
	// Enable or Disable Log output
	KSJ_API  KSJ_LogSet(bool bEnable, const TCHAR *pszFolder);
	// Get Log Status
	KSJ_API  KSJ_LogGet(bool *pbEnable, TCHAR *pszFolder);

	// Get KSJApi.dll Version, call any time.
	KSJ_API  KSJ_GetVersion( int *pnMaj1,  int *pnMaj2,  int *pnMin1,  int *pnMin2);
	// Get Return String Information. call any time.
	KSJ_API  KSJ_GetErrorInfo(int nErrorNo,  TCHAR pszErrorInfo[256]);
	// Initial KSJApi.dll. This Function don't Open Device, only get connected Device Number.
	KSJ_API  KSJ_Init(void);
	// Uninitial KSJApi.dll. This should call correspond to KSJ_Init.
	KSJ_API  KSJ_UnInit(void);
	// Get Connected Device Number, Must call after KSJ_Init.
	KSJ_API  KSJ_DeviceGetCount();
	// Get Device Information.
	KSJ_API  KSJ_DeviceGetInformation(int nIndex,  unsigned short *pusDeviceType,  int *pnSerials,  unsigned short *pusFirmwareVersion);
	// Get Device Information.
	KSJ_API  KSJ_DeviceGetInformationEx(int nIndex,  unsigned short *pusDeviceType,  int *pnSerials,  unsigned short *pusFirmwareVersion,  unsigned short *pusFpgaVersion);

	// Device Parameter Settings
	enum KSJ_PARAM
	{
		KSJ_EXPOSURE = 0,        // Exposure Time (ms)
		KSJ_RED,                 // Red Gain, for line scan sensor map to seg0, for ccd map to KSJ_VGAGAIN
		KSJ_GREEN,               // Green Gain, for line scan sensor map to seg1, for ccd map to KSJ_LAMPLEVEL
		KSJ_BLUE,                // Blue Gain, for CCD map to KSJ_CDSGAIN
		KSJ_GAMMA,               // Gamma
		KSJ_PREVIEW_COLUMNSTART, // Preview Col Start
		KSJ_PREVIEW_ROWSTART,    // Preview Row Start
		KSJ_CAPTURE_COLUMNSTART, // Capture Col Start
		KSJ_CAPTURE_ROWSTART,    // Capture Row Start
		KSJ_HORIZONTALBLANK,     // Horizontal Blank
		KSJ_VERTICALBLANK,       // Vertical Blank
		KSJ_FLIP,                // Flip
		KSJ_BIN,                 // Binning
		KSJ_MIRROR,              // Mirror
		KSJ_CONTRAST,            // Contrast
		KSJ_BRIGHTNESS,          // Brightness
		KSJ_VGAGAIN,             // VGA Gain(CCD)
		KSJ_CLAMPLEVEL,          // Clamp Level(CCD)
		KSJ_CDSGAIN,             // CDS Gain(CCD)
		KSJ_RED_SHIFT,           // Not Use
		KSJ_GREEN_SHIFT,         // Not Use
		KSJ_BLUE_SHIFT,          // Not Use
		KSJ_COMPANDING,          // Companding
		KSJ_EXPOSURE_LINES,      // Exposure Lines
		KSJ_SATURATION,          // Saturation
		KSJ_TRIGGERDELAY,              // Trigger Delay Step = 100uS
		KSJ_STROBEDELAY,               // Not Use
		KSJ_TRIGGER_MODE,              // Trigger Mode
		KSJ_TRIGGER_METHOD,            // Trigger Method
		KSJ_BLACKLEVEL,                // Black Level
		KSJ_BLACKLEVEL_THRESHOLD_AUTO, // Black Level Threshold Auto
		KSJ_BLACKLEVEL_THRESHOLD_LO,   // Black Level Low Threshold
		KSJ_BLACKLEVEL_THRESHOLD_HI    // Black Level High Threshold
	};

	// Get Settings Range
	KSJ_API   KSJ_GetParamRange(int nIndex, KSJ_PARAM Param,  int *pnMinValue,  int *pnMaxValue);
	// Set Parameter
	KSJ_API   KSJ_SetParam(int nIndex, KSJ_PARAM Param,  int nValue);
	// Get Current Parameter Value
	KSJ_API   KSJ_GetParam(int nIndex, KSJ_PARAM Param,  int *pnValue);

	// Set Exposure Time (ms) for float format.( more precision )
	KSJ_API   KSJ_ExposureTimeSet(int nIndex, float fExpTimeMs);
	// Get Exposure Time (ms) for float format.
	KSJ_API   KSJ_ExposureTimeGet(int nIndex,  float *pfExpTimeMs);

    // Skip Mode (Address Mode)
	enum KSJ_ADDRESSMODE
	{
		KSJ_SKIPNONE = 0,
		KSJ_SKIP2,
		KSJ_SKIP3,
		KSJ_SKIP4,
		KSJ_SKIP8
	};

	///////////////////////////////////////////Capture/////////////////////////////////
	// Get The Max Capture FOV( Field Of View ） supported by device
	KSJ_API  KSJ_CaptureGetDefaultFieldOfView(int nIndex, int *pnColStart, int *pnRowStart, int *pnColSize, int *pnRowSize, KSJ_ADDRESSMODE *pAmCol, KSJ_ADDRESSMODE *pAmRow);
	// Set Capture FOV/AOI( Field Of View / Area Of Interesting ）
	KSJ_API  KSJ_CaptureSetFieldOfView(int nIndex, int nColStart, int nRowStart, int nColSize, int nRowSize, KSJ_ADDRESSMODE ColAddressMode, KSJ_ADDRESSMODE RowAddressMode);
	// Get Current Capture FOV / AOI( Field Of View / Area Of Interesting ）
	KSJ_API  KSJ_CaptureGetFieldOfView(int nIndex, int *pnColStart, int *pnRowStart, int *pnColSize, int *pnRowSize, KSJ_ADDRESSMODE *pAmCol, KSJ_ADDRESSMODE *pAmRow);
	// Get Capture Image's Width and Height in Pixel Unit.
	KSJ_API  KSJ_CaptureGetSize(int nIndex,  int *pnWidth,  int *pnHeight);
	// Get Capture Image's Width, Height in Pixel Unit and BitCount( 8, 24, 32 )
	KSJ_API  KSJ_CaptureGetSizeEx(int nIndex, int *pnWidth, int *pnHeight, int *pnBitCount);
	// Capture Raw Data. ( Note: When Set 16bits, Buffer Should double )
	KSJ_API  KSJ_CaptureRawData(int nIndex,  unsigned char *pRawData);
	// Capture 8, 24, 32 Format Image Data.( Note: When Set 16bits, Buffer Should double )
	KSJ_API  KSJ_CaptureRgbData(int nIndex, unsigned char *pRgbData);
	// Capture 8, 24, 32 Format Image Data.( Note: When Set 16bits, Buffer Should double )
	KSJ_API  KSJ_CaptureRgbDataEx(int nIndex, unsigned char *pRgbData, int *pnWidth, int *pnHeight, int *pnBitCount);
	// Save to Bmp Format
	KSJ_API   KSJ_HelperSaveToBmp(unsigned char *pData, int nWidth, int nHeight, int nBitCount, const TCHAR *pszFileName);
	// Save to Jpg Format
	KSJ_API  KSJ_HelperSaveToJpg(unsigned char *pData, int nWidth, int nHeight, int nBitCount, int nQulity, const TCHAR *pszFileName);

	///////////////////////////////////////////Preview/////////////////////////////////
	// Get The Max Preview FOV( Field Of View ） supported by device
	KSJ_API   KSJ_PreviewGetDefaultFieldOfView(int nIndex, int *pnColStart, int *pnRowStart, int *pnColSize, int *pnRowSize, KSJ_ADDRESSMODE *pAmCol, KSJ_ADDRESSMODE *pAmRow);
	// Set Preview FOV/AOI( Field Of View / Area Of Interesting ）
	KSJ_API   KSJ_PreviewSetFieldOfView(int nIndex, int nColStart, int nRowStart, int nColSize, int nRowSize, KSJ_ADDRESSMODE ColAddressMode, KSJ_ADDRESSMODE RowAddressMode);
	// Get Current Preview FOV / AOI( Field Of View / Area Of Interesting ）
	KSJ_API   KSJ_PreviewGetFieldOfView(int nIndex, int *pnColStart, int *pnRowStart, int *pnColSize, int *pnRowSize, KSJ_ADDRESSMODE *pAmCol, KSJ_ADDRESSMODE *pAmRow);
	// Get Preivew Image's Width and Height in Pixel Unit.
	KSJ_API   KSJ_PreviewGetSize(int nIndex, int *pnWidth, int *pnHeight);
	// Get Preview Image's Width, Height in Pixel Unit and BitCount( 8, 24, 32 )
	KSJ_API   KSJ_PreviewGetSizeEx(int nIndex, int *pnWidth, int *pnHeight, int *pnBitCount);

	// Set Preview Position
	KSJ_API   KSJ_PreviewSetPos(int nIndex, HWND hWnd, int x, int y, int nWidth, int nHeight);
	// Get Preview Position
	KSJ_API   KSJ_PreviewGetPos(int nIndex, int *px, int *py, int *pnWidth, int *pnHeight);
	// Preview Mode
	enum KSJ_PREVIEWMODE
	{
		PM_RAWDATA,
		PM_RGBDATA
	};
	// Set Preview Mode
	KSJ_API   KSJ_PreviewSetMode(int nIndex, KSJ_PREVIEWMODE PreviewMode);
	// Get Preview Mode
	KSJ_API   KSJ_PreviewGetMode(int nIndex, KSJ_PREVIEWMODE *pPreviewMode);
	// Get Default Mode
	KSJ_API   KSJ_PreviewGetDefaultMode(int nIndex, KSJ_PREVIEWMODE *pPreviewMode);
	// Start or Stop Preview
	KSJ_API   KSJ_PreviewStart(int nIndex, bool bStart);
	// Pause or Resume Preview
	KSJ_API   KSJ_PreviewPause(int nIndex, bool bPause);
	enum    KSJ_PREVIEWSTATUS
	{
		PS_STOP,
		PS_START,
		PS_PAUSE
	};
	// Get Preview Status
	KSJ_API   KSJ_PreviewGetStatus(int nIndex, KSJ_PREVIEWSTATUS *pPreviewStatus);
	// Get Preview Frame Rate
	KSJ_API   KSJ_PreviewGetFrameRate(int nIndex, float *fFrameRate);
	// Get Paused Image Data when PS_PAUSE Status
	KSJ_API   KSJ_PreviewGetPausedData(int nIndex, unsigned char *pData);
	// Get Paused Image Data when PS_PAUSE Status
	KSJ_API   KSJ_PreviewGetPausedDataEx(int nIndex, unsigned char *pData, int *pnWidth, int *pnHeight, int *pnBitCount);

	// Captrue and Show may Parallel
	KSJ_API  KSJ_PreviewStartEx(int nIndex, bool bStart, bool bParallel);

	KSJ_API  KSJ_PreviewGetStatusEx(int nIndex, KSJ_PREVIEWSTATUS *pPreviewStatus, bool *pbParallel);

	// Get Capture and Show Frame rate when Preview Parallel.
	KSJ_API  KSJ_PreviewGetFrameRateEx(int nIndex, float *fFrameRateCapture, float *fFrameRateShow);
#ifdef linux



#endif


#ifdef __cplusplus
}
#endif

#endif
