; Inno Setup 6 Script for VideoContextEngine Desktop Application
; Per-user installer requiring lowest privileges (no elevation required)

#define MyAppName "VideoContextEngine"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "VideoContextEngine Team"
#define MyAppURL "https://github.com/AshwiniNayak1502/multimodal_video_retrieval"
#define MyAppExeName "VideoContextEngine.exe"

[Setup]
; Unique GUID for VideoContextEngine application
AppId={{5B29D38C-9F8C-4E7B-B963-3F6D82A2E691}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}

; Install per-user into LocalAppData\Programs (requires lowest privileges)
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Output configuration
OutputDir=..\Output
OutputBaseFilename=VideoContextEngine_Setup_x64
SetupIconFile=..\assets\icon.ico
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern

; Architecture: 64-bit Windows only
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; UI Polish
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} Desktop Application
VersionInfoProductName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce

[Files]
; Package the entire PyInstaller onedir distribution
Source: "..\dist\VideoContextEngine\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\icon.ico"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// Check if Microsoft Edge WebView2 Evergreen Runtime is installed on the target machine
function IsWebView2Installed(): Boolean;
var
  InstalledVersion: String;
begin
  Result := False;
  // Check 64-bit / 32-bit registry keys for WebView2 Runtime
  if RegQueryStringValue(HKEY_LOCAL_MACHINE, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-F38F-4475-B9A1-AA05C3C668EC}', 'pv', InstalledVersion) then
  begin
    if (InstalledVersion <> '') and (InstalledVersion <> '0.0.0.0') then
      Result := True;
  end;
  if not Result and RegQueryStringValue(HKEY_CURRENT_USER, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-F38F-4475-B9A1-AA05C3C668EC}', 'pv', InstalledVersion) then
  begin
    if (InstalledVersion <> '') and (InstalledVersion <> '0.0.0.0') then
      Result := True;
  end;
end;

function InitializeSetup(): Boolean;
var
  ErrorCode: Integer;
begin
  Result := True;
  if not IsWebView2Installed() then
  begin
    if MsgBox('Microsoft Edge WebView2 Runtime was not detected on your system.' + #13#10#13#10 +
              'VideoContextEngine requires WebView2 to display its user interface.' + #13#10 +
              'Would you like to open the Microsoft download page now?',
              mbConfirmation, MB_YESNO) = IDYES then
    begin
      ShellExec('open', 'https://go.microsoft.com/fwlink/p/?LinkId=2124703', '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
    end;
  end;
end;

// Preserve user databases, uploads, models, and vectors on uninstall
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    // Note: %LOCALAPPDATA%\VideoContextEngine is deliberately NOT removed
    // to preserve all indexed videos, vector databases, and user accounts.
  end;
end;

