; 忆云 Agent — Inno Setup 安装脚本
;
; 产出：dist\installer\YiyunAgent-0.1.0-win64-setup.exe
; 编译：ISCC.exe installer\YiyunAgent.iss   （或跑 build_installer.bat）
;
; 安装模式：**每用户安装**（PrivilegesRequired=lowest），装到
;   %LOCALAPPDATA%\Programs\YiyunAgent\
; 选这个而不是 Program Files 是为了让「.env / data 放程序旁边」的现有模型
; 继续成立——那个目录当前用户可写，改设置、建库都不用提权，也省掉一套
; 用户数据目录的迁移逻辑。VS Code、GitHub Desktop 都是这个模式。
;
; 需要先跑 build_desktop.bat 生成 dist\YiyunAgent\（本脚本只打包，不负责构建）。

#define MyAppName "忆云 Agent"
#define MyAppNameEn "YiyunAgent"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "wa-nt"
#define MyAppURL "https://github.com/wa-nt/YiyunAgent"
#define MyAppExeName "YiyunAgent.exe"

[Setup]
AppId={{7C3A9E42-5B18-4D6F-9A21-8E4C1F2D6B03}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={localappdata}\Programs\{#MyAppNameEn}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; 每用户安装：不弹 UAC，装到可写的 LOCALAPPDATA
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename={#MyAppNameEn}-{#MyAppVersion}-win64-setup
SetupIconFile=..\web\app.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableDirPage=auto

[Languages]
; 只用 Inno 自带的英文消息：中文消息文件（ChineseSimplified.isl）不在 Inno 默认安装里，
; 引它会让编译直接失败。安装器本身只有几个按钮，英文够用。
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："

[Files]
; 整个 PyInstaller 产物（exe + _internal + skills）。
; **不发 data\ 与 .env**：那是本机运行态和密钥，data\ 里还可能有真实知识库。
; 空的 data\ 由 [Dirs] 建出来。
Source: "..\dist\{#MyAppNameEn}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\{#MyAppNameEn}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\dist\{#MyAppNameEn}\skills\*"; DestDir: "{app}\skills"; Flags: ignoreversion recursesubdirs createallsubdirs
; .env.example 作为配置模板
Source: "..\.env.example"; DestDir: "{app}"; DestName: ".env.example"; Flags: ignoreversion skipifsourcedoesntexist

[Dirs]
; 建出空的运行时目录骨架（数据本身不打包）。用户首次运行前就能看到数据存哪。
Name: "{app}\data"; Permissions: users-modify
Name: "{app}\data\backups"; Permissions: users-modify
Name: "{app}\data\uploads"; Permissions: users-modify

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "立即启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent
