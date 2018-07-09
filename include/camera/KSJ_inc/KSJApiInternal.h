#ifndef __KSJ_API_INTERNAL_H__
#define __KSJ_API_INTERNAL_H__

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

	// 发送控制命令，各字节符合USB协议
	// MSB( IoBlockEx.uOffset ) = SETUPDAT[3], LSB( IoBlockEx.uOffset ) =  SETUPDAT[2] 
	// MSB( IoBlockEx.uIndex )  = SETUPDAT[5], LSB( IoBlockEx.uIndex )  =  SETUPDAT[4] 
	// MSB( IoBlockEx.uLength ) = SETUPDAT[7], LSB( IoBlockEx.uLength ) =  SETUPDAT[6] 
	// wValue  = uOffset
	// wIndex  = uIndex
	// wLength = uLength
#pragma pack ( 1 )
	typedef struct _tagUSBCmd
	{
		unsigned char  bmRequest;
		unsigned char  bRequest;
		unsigned short wValue;
		unsigned short wIndex;
		unsigned short wLength;					// 表示发送或接受数据的长度
	}USBCMD, *PUSBCMD;
#pragma pack ( 8 )

	KSJ_API   KSJ_VendorCmd(int nIndex, PUSBCMD pUsbCmd, unsigned char *pBuf);

	// nIndex is the index of DeviceList in KSJApi internally, not driver symbol name index. It will open first device which not opened as DeviceList[nIndex]
	KSJ_API   KSJ_Open(int nIndex);
	// nIndex is the index of DeviceList in KSJApi internally
	KSJ_API   KSJ_Close(int nIndex);


#ifdef __cplusplus
}
#endif

#endif