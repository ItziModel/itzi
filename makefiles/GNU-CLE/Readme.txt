INSTRUCTIONS FOR COMPILING THE COMMAND LINE VERSION OF SWMM 5
            USING THE GNU C/C++ COMPILER ON LINUX
=====================================================================

1. Open the file swmm5.c in a text editor and make sure that the
   compiler directives at the top of the file read as follows:
       #define CLE
       //#define SOL
       //#define DLL

2. Copy the file "Makefile" to the directory where the SWMM 5 engine
   source code files are located.

3. Open a console shell and navigate to the SWMM 5 engine source
   code directory.

4. Issue the command:

      make

   to create an executable named swmm5.

