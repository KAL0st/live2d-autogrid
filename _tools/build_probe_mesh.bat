@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set REFORGE=D:\LIACF3~1
set BUILD=%REFORGE%\build-nb1012
set QT=D:\Qt\6.8.3\msvc2022_64
set OUT=D:\live2d-autogrid\_tools
set SRC=%OUT%\probe_mesh.cpp

cl /nologo /EHsc /std:c++17 /O2 /MD /W3 /utf-8 ^
  /I"%REFORGE%\src" ^
  /I"%REFORGE%\third_party\miniz" ^
  /I"%QT%\include" ^
  /I"%QT%\include\QtCore" ^
  /DQT_CORE_LIB /DQT_NO_DEBUG /DNDEBUG ^
  "%SRC%" ^
  /Fe:"%OUT%\probe_mesh.exe" ^
  /Fo:"%OUT%\probe_mesh.obj" ^
  /link /LIBPATH:"%BUILD%" /LIBPATH:"%QT%\lib" ^
  l2dcmo3.lib l2dcore.lib miniz.lib Qt6Core.lib

if errorlevel 1 exit /b 1
echo BUILD_OK
