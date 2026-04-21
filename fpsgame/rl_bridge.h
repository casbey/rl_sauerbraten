#pragma once
#define WIN32_LEAN_AND_MEAN
#ifndef WINVER
#define WINVER 0x0600
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdio.h>
#pragma comment(lib, "ws2_32.lib")

namespace rl_bridge
{
    static SOCKET sock = INVALID_SOCKET;
    static bool initialized = false;

    struct RLAction
    {
        float move_fb;   // forward/back
        float strafe;    // strafe
        float yaw_d;     // yaw delta
        float shoot;     // shoot
        bool  reset;     
        bool  valid;     
    };

    inline void init()
    {
        if(initialized) return;
        WSADATA wsa;
        WSAStartup(MAKEWORD(2,2), &wsa);
        sock = socket(AF_INET, SOCK_STREAM, 0);

        DWORD timeout = 5000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));

        sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(42000);
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        if(connect(sock, (sockaddr*)&addr, sizeof(addr)) != 0)
        {
            printf("error");
            closesocket(sock);
            sock = INVALID_SOCKET;
        }
        else printf("connected");
        initialized = true;
    }

    inline bool connected() { return sock != INVALID_SOCKET; }

    inline RLAction exchange(const char *json)
    {
        RLAction act = {0.f, 0.f, 0.f, 0.f, false, false};
        if(!connected()) return act;

        int len = strlen(json);
        if(send(sock, json, len, 0) <= 0)
        {
            closesocket(sock);
            sock = INVALID_SOCKET;
            initialized = false;
            return act;
        }

        char buf[256] = {};
        int n = recv(sock, buf, sizeof(buf)-1, 0);
        if(n <= 0)
        {
            closesocket(sock);
            sock = INVALID_SOCKET;
            initialized = false;
            return act;
        }
        buf[n] = 0;
        act.valid = true;

        // check for reset signal
        if(strstr(buf, "\"reset\""))
        {
            act.reset = true;
            return act;
        }

        // parse continuous action fields
        // format: {"move_fb":0.8,"strafe":-0.3,"yaw_d":0.5,"shoot":0.0}
        auto pf = [&](const char *key) -> float {
            const char *p = strstr(buf, key);
            if(!p) return 0.f;
            p = strchr(p, ':');
            if(!p) return 0.f;
            return (float)atof(p+1);
        };

        act.move_fb = pf("\"move_fb\"");
        act.strafe  = pf("\"strafe\"");
        act.yaw_d   = pf("\"yaw_d\"");
        act.shoot   = pf("\"shoot\"");

        return act;
    }

    inline void shutdown_bridge()
    {
        if(sock != INVALID_SOCKET) { closesocket(sock); sock = INVALID_SOCKET; }
        WSACleanup();
    }
}