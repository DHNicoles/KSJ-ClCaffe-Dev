#ifndef _KSJLOG_H__
#define _KSJLOG_H__

#ifdef _WIN32
#ifdef KSJLOG_EXPORTS                                             // 编译动态链接库时定义
#    pragma message("export dynamic ksjlog lib")
#    define KSJLOG_API __declspec(dllexport) int __stdcall    
#elif  defined KSJLOG_STATIC                                      // 编译静态链接库时，或使用静态链接库时定义
#    pragma message("export or import static ksjlog lib")
#    define KSJLOG_API int __stdcall
#else
#    pragma message("import dynamic ksjlog lib")
#    define KSJLOG_API __declspec(dllimport) int __stdcall           // 默认使用动态链接库
#endif

#elif defined (__GNUC__) && (__GNUC__ >= 4) && (defined (__linux__) || defined (__APPLE__))
#    define KSJLOG_API int
#else
#    error Unknown platform, file needs adaption
#endif

#define		KSJLOG_SUCCESS  0
#define		KSJLOG_FAIL		1

#ifdef __cplusplus
// extern "C"{
#else
#endif



#ifdef linux

typedef void *HANDLE;

#ifndef IN
#    define IN
#endif

#ifndef OUT
#    define OUT
#endif


#ifndef HMODULE
#    define HMODULE int
#endif

#define TCHAR char
#define DWORD short
#define  LONG int
#define TRUE 1
#define FALSE 0
#define _T(x) x

#define BOOL unsigned char


#endif




// bFile控制是否写入到文件, 如果不写入到文件, bPrintOut控制是否弹出提示对话框
// 如果bFile和bPrintOut都为TRUE,则优先级bFile>bPrintOut,只写文件，不会弹出对话框。
// 如果bFile=FALSE,bPrintOut=FALSE,则使用OutputDebugString进行输出。
// szFullPathName为Log日志文件的完整路径。
KSJLOG_API KSJLOG_Init  (HANDLE *hLog, TCHAR* szFullPathName, BOOL bFile, BOOL bPrintOut );

// bFile控制是否写入到文件, 如果不写入到文件, bPrintOut控制是否弹出提示对话框
// 如果bFile=FALSE,bPrintOut=FALSE,则使用OutputDebugString进行输出。
// 如果bFile和bPrintOut都为TRUE,则优先级bFile>bPrintOut,只写文件，不会弹出对话框。
// szLogFileName为Log日志文件名。如果szFolder=NULL,在调用者相同目录下自动创建一个名字为KSJLog的文件夹，并创建szLogFileName的日志文件。
// szFolder最后一个字符不需要'\'，应该是文件夹的完整路径。
// 如果文件夹下有同名文件，则打开文件，进行增量添加。
KSJLOG_API KSJLOG_InitEx(HANDLE *hLog, IN HMODULE hModule, TCHAR* szFolder, TCHAR* szLogFileName, BOOL bFile, BOOL bPrintOut);

KSJLOG_API KSJLOG_UnInit(HANDLE *hLog);

// 输出调试信息, nDebugLevel>=0将被输出。
KSJLOG_API KSJLOG_OutputError(HANDLE hLog, DWORD dwLastError, int nDebugLevel);

// 输出调试信息, nDebugLevel>=0将被输出。
KSJLOG_API KSJLOG_OutputLastError(HANDLE hLog, int nDebugLevel);

// nDebugLevel只要大于等于UDBG_LEVEL=0，则进行日志输出，
// 如果bFile = TRUE, 则写文件，
// 如果bPrintOut = TRUE,则弹出对话框，
// 否则用OutputDebugString语句打印
KSJLOG_API KSJLOG_OutputString(HANDLE hLog, const TCHAR szItem[], int nDebugLevel);

KSJLOG_API KSJLOG_OutputStringEx(HANDLE hLog, int nDebugLevel, TCHAR* lpszFormat, ...);

#ifdef __cplusplus
//}
#endif

#endif
