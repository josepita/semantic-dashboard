Set WshShell = CreateObject("WScript.Shell")
' Cambiar al directorio del script
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
' Ejecutar el batch sin mostrar ventana (windowstyle 0 = oculto, 1 = normal, 7 = minimizado)
WshShell.Run "start_streamlit.bat", 0, False
