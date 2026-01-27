Set WshShell = CreateObject("WScript.Shell")
' Cambiar al directorio de la app
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\apps\gsc_insights"
' Ejecutar el batch con ventana minimizada
WshShell.Run "start_app.bat", 7, False
