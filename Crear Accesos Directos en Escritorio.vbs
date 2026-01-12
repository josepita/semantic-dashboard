Option Explicit

Dim objShell, objFSO, strDesktop, strScriptPath, strShortcut
Dim arrApps, app, shortcutName, targetPath

' Crear objetos
Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Obtener ruta del escritorio
strDesktop = objShell.SpecialFolders("Desktop")

' Obtener ruta de este script
strScriptPath = objFSO.GetParentFolderName(WScript.ScriptFullName)

' Array con las aplicaciones a crear accesos directos
arrApps = Array( _
    "Launcher - Dashboard Principal.vbs|SEO Dashboard Principal", _
    "Launcher - Content Analyzer.vbs|SEO Content Analyzer", _
    "Launcher - Linking Optimizer.vbs|SEO Linking Optimizer", _
    "Launcher - GSC Insights.vbs|GSC Insights" _
)

' Crear accesos directos
Dim i, parts, createdCount
createdCount = 0

For i = 0 To UBound(arrApps)
    parts = Split(arrApps(i), "|")
    targetPath = strScriptPath & "\" & parts(0)
    shortcutName = parts(1)

    ' Verificar que el archivo fuente existe
    If objFSO.FileExists(targetPath) Then
        ' Crear acceso directo
        Set strShortcut = objShell.CreateShortcut(strDesktop & "\" & shortcutName & ".lnk")
        strShortcut.TargetPath = targetPath
        strShortcut.WorkingDirectory = strScriptPath
        strShortcut.Description = "Inicia " & shortcutName
        strShortcut.Save

        createdCount = createdCount + 1
    End If
Next

' Mostrar mensaje de éxito
If createdCount > 0 Then
    MsgBox "✅ Se crearon " & createdCount & " accesos directos en el escritorio." & vbCrLf & vbCrLf & _
           "Puedes personalizarlos:" & vbCrLf & _
           "1. Clic derecho → Propiedades" & vbCrLf & _
           "2. Cambiar icono" & vbCrLf & _
           "3. Anclar a barra de tareas", vbInformation, "Accesos Directos Creados"
Else
    MsgBox "❌ No se encontraron archivos launcher." & vbCrLf & _
           "Verifica que los archivos .vbs existen en:" & vbCrLf & strScriptPath, vbExclamation, "Error"
End If

' Limpiar
Set objShell = Nothing
Set objFSO = Nothing
