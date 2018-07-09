#ifndef __KSJ_API_CUSTOM_H__
#define __KSJ_API_CUSTOM_H__

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

	// 将pData(宽度为nWidth, 高度为nHeight, 比特位数为nBitCount)的图像数据缩放为宽度为nStretchWidh, 高度为nStretchHeight的数据并填充到用户分配的pStretchData缓冲区,比特位数不变
	KSJ_API  KSJ_HelperStretchBmp(unsigned char *pData, int nWidth, int nHeight, int nBitCount, unsigned char *pStretchData, int nStretchWidth, int nStretchHeight);
	// 根据图像的比特深度得到BMP图像的文件头,信息头,及颜色表总的字节数目
	KSJ_API  KSJ_HelperGetBmpFileHeaderSize(int nBitCount, int *pnTotalBytes);
	// 得到BMP图像的文件头和信息头
	KSJ_API  KSJ_HelperGetBmpFileHeader(int nWidth, int nHeight, int nBitCount, void *pHeader);
	// 采集一帧图像到系统剪贴板
	KSJ_API  KSJ_CaptureRgbDataToClipboard(int nIndex);

	// 设置FFC视场
	KSJ_API  KSJ_SetFlatFieldCorrectionFieldOfView(int nIndex, int nColStart, int nRowStart, int nColSize, int nRowSize, KSJ_ADDRESSMODE ColAddressMode, KSJ_ADDRESSMODE RowAddressMode, unsigned short wMultiFrameNum);

	// 获取FFC的视场，视场必须相同才可以进行校正。
	KSJ_API  KSJ_GetFlatFieldCorrectionFieldOfView(int nIndex, int *pnColStart, int *pnRowStart, int *pnColSize, int *pnRowSize, KSJ_ADDRESSMODE *pAmCol, KSJ_ADDRESSMODE *pAmRow, unsigned short *pwMultiFrameNum);

	// 根据采集视场采集nDarkFrames张图像，平均后作为DarkFrame
	KSJ_API  KSJ_MakeDarkFrame(int nIndex, int nDarkFrames);
	// 根据采集视场采集nFlatFieldFrames张图像，平均后作为FlatFieldImage
	KSJ_API  KSJ_MakeFlatFieldFrame(int nIndex, int nFlatFieldFrames);

	// 预览时是否使能
	KSJ_API  KSJ_PreviewSetFlatFieldCorrection(int nIndex, bool bEnable);

	KSJ_API  KSJ_PreviewGetFlatFieldCorrection(int nIndex, bool *pbEnable);
	// 采集时是否使能
	KSJ_API  KSJ_CaptureSetFlatFieldCorrection(int nIndex, bool bEnable);

	KSJ_API  KSJ_CaptureGetFlatFieldCorrection(int nIndex, bool *pbEnable);

	KSJ_API  KSJ_SaveFlatFieldCorrection(int nIndex, TCHAR *pszFilePath);

	KSJ_API  KSJ_LoadFlatFieldCorrection(int nIndex, TCHAR *pszFilePath);

#ifdef linux
	KSJ_API   KSJ_SoftStartCapture(int nIndex);
	KSJ_API   KSJ_ReadRawDataAfterStart(int nIndex, unsigned char *pRawData);
	KSJ_API   KSJ_ReadRgbDataAfterStart(int nIndex, unsigned char *pRgbData);
#endif


#ifdef __cplusplus
}
#endif

#endif
