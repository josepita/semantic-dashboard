Set WshShell = CreateObject("WScript.Shell")
' Cambiar al directorio del script
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
' Ejecutar el batch con ventana minimizada (windowstyle 7 = minimizado)
WshShell.Run "start_streamlit.bat", 7, False
