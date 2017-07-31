//-----------------------------------------------------------------------------
//   swmm5.h
//
//   Project: EPA SWMM5
//   Version: 5.1
//   Date:    03/24/14  (Build 5.1.001)
//   Author:  L. Rossman
//
//   Prototypes for SWMM5 functions exported to swmm5.dll.
//
//-----------------------------------------------------------------------------
#ifndef SWMM5_H
#define SWMM5_H

// --- define WINDOWS

#undef WINDOWS
#ifdef _WIN32
  #define WINDOWS
#endif
#ifdef __WIN32__
  #define WINDOWS
#endif

// --- define DLLEXPORT

#ifdef WINDOWS
  #define DLLEXPORT __declspec(dllexport) __stdcall
#else
  #define DLLEXPORT
#endif

// --- use "C" linkage for C++ programs

#ifdef __cplusplus
extern "C" { 
#endif 

int  DLLEXPORT   swmm_run(char* f1, char* f2, char* f3);
int  DLLEXPORT   swmm_open(char* f1, char* f2, char* f3);
int  DLLEXPORT   swmm_start(int saveFlag);
int  DLLEXPORT   swmm_step(double* elapsedTime);
int  DLLEXPORT   swmm_end(void);
int  DLLEXPORT   swmm_report(void);
int  DLLEXPORT   swmm_getMassBalErr(float* runoffErr, float* flowErr,
                 float* qualErr);
int  DLLEXPORT   swmm_close(void);
int  DLLEXPORT   swmm_getVersion(void);

// Coupling functions (GESZ)
int DLLEXPORT   swmm_getNodeID(int index, char* id);
int DLLEXPORT   swmm_getNodeData(int index, nodeData* data);
int DLLEXPORT   swmm_getNodeInflows(double* flows);
int DLLEXPORT   swmm_getNodeOutflows(double* flows);
int DLLEXPORT   swmm_getNodeHeads(double* heads);
int DLLEXPORT   swmm_addNodeInflow(int index, double inflow);
int DLLEXPORT   swmm_getLinkID(int index, char* id);
int DLLEXPORT   swmm_getLinkData(int index, linkData* data);

// Coupling functions (L. Courty)
int DLLEXPORT   swmm_setNodeFullDepth(int index, double depth);
int DLLEXPORT   swmm_setAllowPonding(int ap);
int DLLEXPORT   swmm_setNodePondedArea(int index, double area);

#ifdef __cplusplus 
}   // matches the linkage specification from above */ 
#endif

#endif
