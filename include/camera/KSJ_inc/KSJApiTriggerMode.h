#ifndef KSJAPI_TRIGGERMODE_H__
#define KSJAPI_TRIGGERMODE_H__

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
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////////////////
// Trigger Mode
enum KSJ_TRIGGERMODE
{
	KSJ_TRIGGER_INTERNAL,
	KSJ_TRIGGER_EXTERNAL,
	KSJ_TRIGGER_SOFTWARE,
	KSJ_TRIGGER_FIXFRAMERATE
};

// Set Trigger Mode
KSJ_API  KSJ_TriggerModeSet(int nIndex,  KSJ_TRIGGERMODE TriggerMode);
KSJ_API  KSJ_TriggerModeGet(int nIndex,  KSJ_TRIGGERMODE *pTriggerMode);

enum KSJ_TRIGGERSTATUS
{
	KSJ_NO,
	KSJ_HAVE
};

// Get Frame Buffer Status. Has One or more Frame or None. Maybe return RET_NOTSUPPORT.
// When Fpga Version is not 0.0, This Function should support.
KSJ_API  KSJ_TriggerStatusGet(int nIndex, KSJ_TRIGGERSTATUS *pTriggerStatus);

// Empty Frame Buffer. Not Support When In KSJ_TRIGGER_SOFTWARE Mode.
KSJ_API  KSJ_EmptyFrameBuffer(int nIndex);

// Same to call KSJ_EmptyFrameBuffer before KSJ_CaptureRawData. Not Support When In KSJ_TRIGGER_SOFTWARE Mode.
KSJ_API  KSJ_CaptureRawDataAfterEmptyFrameBuffer(int nIndex, unsigned char *pRawData);
// Same to call KSJ_EmptyFrameBuffer before KSJ_CaptureRgbData. Not Support When In KSJ_TRIGGER_SOFTWARE Mode.
KSJ_API  KSJ_CaptureRgbDataAfterEmptyFrameBuffer(int nIndex, unsigned char *pRgbData);
// Same to call KSJ_EmptyFrameBuffer before KSJ_CaptureRgbDataEx. Not Support When In KSJ_TRIGGER_SOFTWARE Mode.
KSJ_API  KSJ_CaptureRgbDataExAfterEmptyFrameBuffer(int nIndex, unsigned char *pRgbData, int *pnWidth, int *pnHeight, int *pnBitCount);


// Fixed Frame Rate when in KSJ_TRIGGER_FIXFRAMERATE mode. Maybe Return RET_NOTSUPPORT.
KSJ_API  KSJ_SetFixedFrameRateEx(int nIndex, float fFrameRate);
KSJ_API  KSJ_GetFixedFrameRateEx(int nIndex, float *pfFrameRate);

enum KSJ_TRIGGERMETHOD
{
	KSJ_TRIGGER_FALLINGEDGE,
	KSJ_TRIGGER_RISINGEDGE,
	KSJ_TRIGGER_HIGHLEVEL,
	KSJ_TRIGGER_LOWLEVEL
};
// Set Trigger Method when in KSJ_TRIGGER_EXTERNAL mode.
KSJ_API  KSJ_TriggerMethodSet(int nIndex, KSJ_TRIGGERMETHOD  TriggerMethod);
KSJ_API  KSJ_TriggerMethodGet(int nIndex, KSJ_TRIGGERMETHOD *pTriggerMethod);

// Set Trigger Delay when in KSJ_TRIGGER_EXTERNAL. May be Return RET_NOTSUPPORT.
KSJ_API  KSJ_TriggerDelayGetRange(int nIndex, unsigned short *wMin, unsigned short *wMax);
KSJ_API  KSJ_TriggerDelaySet(int nIndex, unsigned short wDelay);
KSJ_API  KSJ_TriggerDelayGet(int nIndex, unsigned short *pwDelay);

// Set Read action timeout (ms). Influence capture and preview.
KSJ_API  KSJ_CaptureSetTimeOut(int nIndex, unsigned int dwTimeOut);
// Get Read action image timeout (ms).
KSJ_API  KSJ_CaptureGetTimeOut(int nIndex, unsigned int *pdwTimeOut);

// Set If recover device when Read Fail. It will Influence capture and preview.
KSJ_API  KSJ_CaptureSetRecover(int nIndex, bool bRecover);
// Get If recover device when Read Fail.
KSJ_API  KSJ_CaptureGetRecover(int nIndex, bool *pbRecover);

// Set timeout and recover at the same time
KSJ_API  KSJ_CaptureSetTimeOutEx(int nIndex, unsigned int dwTimeOut, bool bRecover);
// Get timeout and recover at the same time
KSJ_API  KSJ_CaptureGetTimeOutEx(int nIndex, unsigned int *pdwTimeOut, bool *pbRecover);

#ifdef __cplusplus
}
#endif

#endif