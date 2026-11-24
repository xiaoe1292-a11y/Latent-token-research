# 上传到 GitHub 并展示在个人主页

建议仓库名：`latte-latent-beam-study`。

建议 Description：`Latte reproduction and latent-token × beam-size ablations for generative recommendation, with auditable results and CPU analysis tools.`

建议 Topics：`generative-recommendation`、`recommendation-system`、`semantic-id`、`beam-search`、`pytorch`、`reproducible-research`。

## 网页上传

1. 解压项目 ZIP，进入包含本项目 `README.md` 的目录。
2. 在 GitHub 新建上述名称的空仓库，选择是否公开。不要勾选自动生成 README、LICENSE 或 gitignore，以免与本包重复。
3. 选择 **uploading an existing file / Add file → Upload files**，上传项目文件与目录，然后提交。不要只把 ZIP 当成一个文件上传，否则首页不会直接显示 README 和图片。
4. 确认首页结果矩阵、曲线和 Pareto 图片都正常显示。它们使用相对路径，不需要填写你的用户名。
5. 首次提交后可在 Actions 页查看 CPU 核验工作流。仅本地测试已执行；本包不预先宣称远端 CI 通过。

网页上传可能漏掉隐藏文件或超过文件数量限制；需要完整保留 `.github/`、`.gitignore`、`.gitattributes` 时，推荐下面的 Git 方式。

## 用 Git 上传完整目录

在解压后的项目根目录运行，先将 `<YOUR_USERNAME>` 替换成你的 GitHub 用户名；远端仓库需先创建为空仓库。

```bash
git init -b main
git add .
git commit -m "Publish Latte latent-token and beam-size study"
git remote add origin https://github.com/<YOUR_USERNAME>/latte-latent-beam-study.git
git push -u origin main
```

以上命令仅为上传说明，整理项目时没有替你执行 `git push` 或创建远端仓库。推送前可先运行 README 的 CPU 复核命令。

## 展示到个人主页

仓库上传后，在个人主页选择 **Customize your pins**，勾选这个项目即可展示为置顶项目。普通项目仓库与“用户名同名的个人主页 README 仓库”是两种用途；本包适合作为独立项目仓库，再固定到主页。

如果你已有个人主页 README，可以用下面这段项目简介，并把项目名链接到刚创建的真实仓库地址：

> **Latte 生成式推荐复现与容量/搜索消融**：在 Amazon Reviews 2023 / Video_Games 的 94,762 名测试用户上完成 1/2/4/8/16 latent 消融与 16 组 latent × beam 联合验证，提供 Recall/NDCG、候选覆盖、延迟/吞吐/显存和配对 Bootstrap 分析，以及可复算代码与图表。

项目介绍中保留“固定预算、单训练种子”和“联合实验成本来自共享 GPU”的口径，不写成提出了 Latte、达到 SOTA 或验证出 Scaling Law。
