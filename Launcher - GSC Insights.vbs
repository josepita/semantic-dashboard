Set WshShell = CreateObject("WScript.Shell")
' Cambiar al directorio de la app
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\apps\gsc_insights"
' Ejecutar el batch sin mostrar ventana
WshShell.Run "start_app.bat", 0, False
