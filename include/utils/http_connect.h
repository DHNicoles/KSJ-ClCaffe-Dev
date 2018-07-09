#ifndef http_connect_h__
#define http_connect_h__

#include "utils/util.h"
#ifndef WIN32
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <netdb.h>
#include <stdarg.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>    
#include "utils/singleton.h"
#endif // !WIN32
#define BUFSIZE 41000
#define URLSIZE 1024
#define INVALID_SOCKET -1
namespace ice
{
	class HttpConnect : public Singleton<HttpConnect>
	{
	public:
		HttpConnect();
		~HttpConnect();
		void DebugOut(const char *fmt, ...);
	
		int HttpGet(const char* strUrl, char* strResponse);
		int HttpPost(const char* strUrl, const char* strData, char* strResponse);
	
		char* GetIPFromUrl(const char* strUrl);
	private:
		int   HttpRequestExec(const char* strMethod, const char* strUrl, const char* strData, char* strResponse);
		char* HttpHeadCreate(const char* strMethod, const char* strUrl, const char* strData);
		char* HttpDataTransmit(char *strHttpHead, const int iSockFd);
	
		int   GetPortFromUrl(const char* strUrl);
		char* GetParamFromUrl(const char* strUrl);
		char* GetHostAddrFromUrl(const char* strUrl);
	
		int   SocketFdCheck(const int iSockFd);
	
		static int m_iSocketFd;
	};
}

#endif // http_connect_h__
