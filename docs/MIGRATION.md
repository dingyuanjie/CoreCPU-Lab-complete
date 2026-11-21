# 将实验室代码合入原有 CoreCPU GitHub 仓库

**此压缩包是本地生成的源码快照，不自动修改你的 GitHub 仓库，也不拥有原 Git commit 历史。** 建议先在原仓库创建新分支，将本项目源码复制到工作区并检查差异；不要直接覆盖旧文件。

```bash
# 在你已克隆的旧 CoreCPU 仓库中：
git status
git tag corecpu-v0.1-before-lab
git checkout -b experiment/corecpu-lab
# 再将解压后的 CoreCPU-Lab 内容拷贝到该工作区
# 注意保留你原本已经存在的源码、数据、实验日志等
git add -A
git commit -m "Introduce CoreCPU-Lab factorial architecture experiments"
```

GitHub 仓库重命名请在 GitHub 的 Settings → General → Repository name 进行，或者使用你已经配置身份验证的 GitHub CLI。**无需新建仓库，也不要丢失原仓库 commit、issue、tag 与许可证。**

是否公开由你决定；这个代码包没有为原仓库擅自添加开源许可证。
