INSTRUCTIONS FOR COMPILING SWMM5.DLL USING MICROSOFT VISUAL C++ 2010
=====================================================================

1. Open the file swmm5.c in a text editor and make sure that the
   compiler directives at the top of the file read as follows:
       //#define CLE
       //#define SOL
       #define DLL

2. Create a sub-directory named VC2010_DLL under the directory where
   the SWMM 5 Engine source code files are stored and copy SWMM5.DEF
   and VC2010-DLL.VCPROJ to it.

3. Launch Visual C++ 2010 and use the File >> Open command to open
   the VC2010-DLL.VCPROJ file.

4. Issue the Build >> Configuration Manager command and select the
   Release configuration.

5. Issue the Build >> Build VC2010-DLL command to build SWMM5.DLL
   (which will appear in the Release subdirectory underneath the
   VC2010-DLL directory).

NOTE: The VC-2010 project file includes Open MP support which is
      only available with the Professional and higher versions of
      the compiler.   
