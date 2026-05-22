# 将本地 main 分支推送到 GitHub（需能访问 github.com:443）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .git)) {
    Write-Error "当前目录不是 Git 仓库，请先在 d:\Unimiss\single 初始化。"
}

$remote = "https://github.com/Baiyouawa/Unimiss-single-.git"
git remote get-url origin 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    git remote add origin $remote
} else {
    git remote set-url origin $remote
}

Write-Host "正在推送到 $remote ..."
# 全局 gitconfig 中 credential.helper 为空，此处临时启用 Windows 凭据管理器
git -c credential.helper=manager push -u origin main
if ($LASTEXITCODE -eq 0) {
    Write-Host "推送成功: https://github.com/Baiyouawa/Unimiss-single-"
} else {
    Write-Host @"

推送失败。常见原因：
1. 无法访问 GitHub（校园网/防火墙）— 请开 VPN 或配置代理后重试
2. 未登录 GitHub — 推送时会弹出凭据窗口，或到 GitHub 添加 SSH 公钥后用 SSH 地址推送
3. 仓库不存在或无权限 — 确认已登录账号 Baiyouawa 且仓库已创建

SSH 方式（需先在 GitHub → Settings → SSH keys 添加公钥）：
  git remote set-url origin git@github.com:Baiyouawa/Unimiss-single-.git
  git push -u origin main
"@
    exit 1
}
