<[Git 教程 | 菜鸟教程](https://www.runoob.com/git/git-tutorial.html)>

以下是需要掌握的核心 Git 知识点，分为**基础必备**、**进阶核心**和**高级与最佳实践**三个层面。

---

### 一、 基础必备：日常操作与生存技能

这些命令是你每天都会用到的，是进行任何工作的基础。

1.  **仓库克隆与初始化**
    *   `git clone <url>`：**最重要的第一步**。从远程仓库（如 GitHub, GitLab）获取代码到本地。
    *   `git init`：将本地目录初始化为 Git 仓库。

2.  **查看状态与历史**
    *   `git status`：查看工作区和暂存区的状态。**有任何疑惑就先跑一下这个命令**。
    *   `git log`：查看提交历史。常用选项：
        *   `--oneline`：简洁模式显示。
        *   `--graph`：图形化显示分支合并历史。
        *   `-p`：显示每次提交的具体内容差异。

3.  **文件操作与提交**
    *   `git add <file>` 或 `git add .`：将文件更改添加到暂存区。
    *   `git commit -m "commit message"`：将暂存区的更改提交到本地仓库。
        *   **Commit Message 规范**：对于科研，清晰的提交信息至关重要。例如：
            *   `feat: add ResNet-50 model definition`
            *   `fix: correct gradient calculation in loss function`
            *   `exp: experiment with learning rate 1e-4 on dataset A`
    *   `git restore <file>`：丢弃工作区的更改（Git 2.23+）。
    *   `git rm <file>`：从 Git 和文件系统中删除文件。

4.  **与远程仓库交互**
    *   `git pull`：从远程仓库拉取更新并合并到当前分支（=`git fetch` + `git merge`）。
    *   `git push`：将本地提交推送到远程仓库。
    *   `git remote -v`：查看远程仓库地址。

---

### 二、 进阶核心：应对复杂场景与协作

这部分是高效管理和复现实验的关键。

1.  **分支管理 - 核心中的核心**
    *   `git branch`：查看所有分支。
    *   `git branch <branch_name>`：创建新分支。
    *   `git checkout <branch_name>` 或 `git switch <branch_name>`：切换分支。
    *   `git checkout -b <new_branch>`：创建并切换到新分支。**这是为每个新实验（新模型、新参数）创建独立环境的标准操作**。
    *   `git merge <branch_name>`：将指定分支合并到当前分支。
    *   `git branch -d <branch_name>`：删除已合并的分支。

2.  **比较差异**
    *   `git diff`：比较**工作区**和**暂存区**的差异。
    *   `git diff --staged`：比较**暂存区**和**最后一次提交**的差异。
    *   `git diff <commit_A> <commit_B>`：比较两次提交之间的差异。**用于分析不同实验结果对应的代码变化**。

3.  **撤销与回退 - “救命”命令**
    *   `git restore --staged <file>`：将文件从暂存区撤回到工作区（unstage）。
    *   `git commit --amend`：修改最后一次提交的信息或内容（**注意：不要对已推送的提交使用**）。
    *   `git reset --hard <commit_hash>`：**危险命令**。将工作区和暂存区彻底回退到某次提交的状态。常用于放弃当前所有实验性更改。
    *   `git revert <commit_hash>`：**安全命令**。创建一个新的提交来“撤销”某次提交的更改。推荐用于撤销已推送到远程的提交。

4.  **暂存更改**
    *   `git stash`：将当前工作区和暂存区的修改临时保存起来，让工作区变干净。
    *   `git stash pop`：恢复最近一次暂存的修改。
    *   **使用场景**：当你正在修改代码，需要紧急切换到另一个分支（如 `main`）去拉取更新或修复 bug 时。

---

### 三、 高级与最佳实践：提升效率与代码质量

这些技能让你看起来像个专家，并能有效应对论文复现中的复杂问题。

1.  **理解 Git 工作流**
    *   **功能分支工作流**：为每个新功能（新模型架构、新数据集）创建一个独立分支，开发测试完成后合并回 `main` 或 `develop` 分支。这是最适用于科研的流程。

2.  **处理合并冲突**
    *   当多人修改了同一文件的同一部分时，`git merge` 或 `git pull` 可能会产生冲突。
    *   你需要手动编辑文件中被 `<<<<<<<`， `=======`， `>>>>>>>` 标记的部分，解决冲突后，执行 `git add` 和 `git commit` 来完成合并。

3.  **`.gitignore` 文件**
    *   一个至关重要的文件，用于告诉 Git 忽略哪些文件或目录。
    *   **在科研中必须忽略**：大型数据集、模型检查点（`.ckpt`, `.pth`）、TensorBoard/Python 缓存（`__pycache__/`）、日志文件、系统文件（`.DS_Store`）等。这能保持仓库的清洁和小巧。

4.  **查看特定代码的历史**
    *   `git blame <file_name>`：逐行显示文件，并标注每一行是谁在什么时候修改的。**用于理解某段关键代码（如模型核心层）的演变过程**。

5.  **子模块与Fork工作流（常见于论文复现）**
    *   **Fork**：在 GitHub 等平台上，将别人的仓库复制到自己账户下。这是你复现论文代码的起点。
    *   **`git submodule`**：允许你将一个 Git 仓库作为另一个仓库的子目录。常用于管理依赖项（如某个特定的模型库）。
        *   `git submodule add <url>`：添加子模块。
        *   `git clone --recurse-submodules <url>`：克隆包含子模块的仓库。

### 总结：典型工作流示例

**场景：你找到一篇论文的代码，想要复现并尝试自己的改进。**

1.  **获取代码**：`git clone https://github.com/author/paper-code.git`
2.  **进入目录**：`cd paper-code`
3.  **为你的实验创建分支**：`git checkout -b my_experiment`
4.  **进行代码修改、调参、训练**。
5.  **随时提交**：
    *   `git add .`
    *   `git commit -m "exp: change optimizer to AdamW"`
6.  **（如果需要）从原仓库同步更新**：
    *   `git checkout main` // 切换回主分支
    *   `git pull origin main` // 拉取原作者的最新更新
    *   `git checkout my_experiment` // 切换回你的实验分支
    *   `git merge main` // 将更新合并到你的分支，解决可能出现的冲突
7.  **实验完成，整理代码**：可能需要 `git rebase -i` 来整理提交历史，使其清晰易懂。
8.  **推送到你自己的远程仓库**（如果你 Fork 了的话）: `git push -u origin my_experiment`
9.  **发起 Pull Request**：在 GitHub 上向原仓库提交你的改进（可选）。
