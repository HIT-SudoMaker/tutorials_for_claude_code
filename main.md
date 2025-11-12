<h1 align="center">What could a soul like that be capable of ?</h1>
<h1 align="center">基于Zero-Config Code Flow的Claude Code配置教程</h1>
---

**作者**：劳家康 & 韩律元
**单位**：哈尔滨工业大学，仪器科学与工程学院 & 未来技术学院
**日期**：2025年11月11日

---

## 背景介绍

使用自然语言实现程序设计和开发，一直是无数“编程小白”的终极梦想，也是学术界和工业界长期努力的研究方向。近年来，大语言模型（Large Language Models，LLMs）的快速发展，推动了AI驱动的开发环境（AI-Driven Development Environments，AIDEs）的长足进步。以[Cursor](https://cursor.com/cn)和集成[GitHub Copilot](https://github.com/features/copilot)的[Visual Studio Code](https://code.visualstudio.com)为代表的AIDEs，已经让这一梦想部分实现，通过智能代码补全和自动化辅助，提升了编程效率。

然而，当广大学生用户将这些工具应用于实际科研编程任务时，往往面临成本、时间和性能的限制。一方面以Cursor为例，其全代码库上下文感知能力和代码补全体验备受好评，但其频繁变更的付费策略对学生用户并不友好；另一方面以GitHub Copilot为例，其通过[GitHub Education](https://education.github.com)计划为学生用户提供免费使用资格，但其相对较弱的上下文感知能力和代码补全体验，在处理复杂项目需求时往往力不从心。除此之外，市场上的其他选项，例如以[Trae](https://www.trae.cn)、[Qoder](https://qoder.com)、[CodeBuddy](https://copilot.tencent.com)为代表的新兴AIDEs，虽然性价比颇高，但在模型切换灵活性和代码补全体验上仍然存在短板；而以[Continue](https://www.continue.dev)、[Roo Code](https://www.continue.dev)为代表的开源插件，虽然提供高度定制性，但也因此提高了使用门槛，难以实现开箱即用的效果。

以上问题促使了[Anthropic](https://www.anthropic.com)的[Claude Code](https://claude.com/product/claude-code)和[OpenAI](https://openai.com)的[Codex](https://openai.com/codex)等基于命令行界面（Command Line Interface，CLI）的下一代编程智能体（Agents）的出现。这些智能体不仅通过插件框架、子代理协作、工具调用和沙盒执行等技术手段，显著提升了自主编程能力，甚至可以集成多种模型上下文协议（Model Context Protocols，MCPs），为用户提供更为先进的程序设计和开发体验。当然，使用这些先进的智能体对于中国地区的学生用户而言也并非易事。以Claude Code为例，由于Anthropic官方高昂的资费和不友好的地区政策，诸多用户必须寻求“曲线救国”的方案，一是需要通过配置开源项目[Claude Code Router（CCR）](https://github.com/musistudio/claude-code-router)实现对Claude Code模型的应用程序编程接口（Application Programming Interface，API）代理，二是需要集成其他MCPs以弥补代理模型相对于官方模型的性能差异。这对于本来就是“编程小白”的学生用户而言，无疑是一道难以逾越的门槛。

为彻底解决以上问题，本教程从广大学生用户的需求出发，在Windows操作系统下基于开源项目[Zero-Config Code Flow（ZCF）](https://github.com/UfoMiao/zcf)的一站式安装和设置方案，详细介绍如何以极低的成本和时间投入，快速搭建起一个高性能的自动化编程环境，充分释放现代AI工具的强大生产力。

---

## 安装ZCF

### 安装Node.js环境

根据ZCF的项目说明，安装ZCF需要使用指令：

```bash
npx zcf
```

其中，npx是[Node.js](https://nodejs.org/zh-cn)环境下的包执行工具，因此需要下载其最新长期支持（Long Term Support，LTS）版本的安装包。

<div style="text-align: center;">
  <img src="figures/figure1.png" alt="Node.js下载页面" width="70%"/>
  <br />
  <a href="https://nodejs.org/zh-cn/download">Node.js下载页面链接：https://nodejs.org/zh-cn/download</a>
</div>

下载完成后，双击安装包文件进行安装，Node.js环境的安装过程较为简单，除了需要勾选“I accept the terms in the License Agreement”选项外，建议全部保持默认设置。

### 安装Git Bash终端

尽管根据ZCF的项目说明，安装ZCF并不需要预安装Git环境，但是编者强烈建议Windows操作系统的用户安装Git Bash终端。这是因为Git Bash终端内置了可移植操作系统接口（Portable Operating System Interface，POSIX）兼容环境，可以显著提升Claude Code性能的稳定性，同时也为日后的代码仓库版本管理提供了便利。因此，建议访问其官网下载最新版本的安装包。

<div style="text-align: center;">
  <img src="figures/figure2.png" alt="Git Bash终端下载页面" width="70%"/>
  <br />
  <a href="https://git-scm.com/install/windows">Git Bash终端下载页面链接：https://git-scm.com/install/windows</a>
</div>

下载完成后，双击安装包文件进行安装，Git Bash终端的安装过程同样简单，除了需要勾选“Add a Git Bash Profile to Windows Terminal”选项和“Use Visual Studio Code as Git's default editor”选项外，建议全部保持默认设置。

需要说明的是，ZCF和Claude Code完全支持原生Windows PowerShell终端，下文的所有操作均以Windows PowerShell终端为例，无需额外配置。若读者偏好类Unix风格的终端，可以全程使用Git Bash终端执行命令，两种终端的使用方法和效果完全一致。

### 安装ZCF并执行完整初始化

成功安装Node.js环境和Git Bash终端后，打开Windows PowerShell终端，输入以下命令安装ZCF：

```bash
npx zcf
```

但是此时往往会遭遇如下错误提示：

```bash
npx : 无法加载文件 C:\Program Files\nodejs\npx.ps1，因为在此系统上禁止运行脚本。有关详细信息，请参阅 https:/go.microsoft.com/fwlink/?LinkID=135170 中的 about_Execution_Policies。
```

<div style="text-align: center;">
  <img src="figures/figure3.png" alt="PowerShell终端执行策略错误提示" width="70%"/>
  <br />
  PowerShell终端执行策略错误提示
</div>

这是因为在默认情况下，Windows PowerShell终端禁止执行未签名的脚本文件。因此需要修改Windows PowerShell终端的执行策略：

```bash
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

修改执行策略后，重新打开Windows PowerShell终端，即可通过上文的命令成功安装ZCF。

<div style="text-align: center;">
  <img src="figures/figure4.png" alt="ZCF主页面" width="70%"/>
  <br />
  ZCF主页面
</div>

安装ZCF后，需要执行完整初始化流程，在主页面中输入选项“1”并回车开始初始化，推荐具体设置如下：

- **选择Claude Code配置语言**：推荐“English”选项

- **选择AI输出语言**：推荐“简体中文”选项

- **选择API配置模式**：推荐“使用CCR代理”选项

- **选择要安装的工作流类型**：推荐全选

- **选择要安装的输出风格**：推荐“专业的软件工程师”选项

- **选择要安装的输出风格**：推荐“工程师专业版”和“默认风格”选项

- **是否配置MCP服务**：推荐“Y”选项

- **选择要安装的MCP服务**：推荐全选除了“Exa AI 搜索”以外的所有选项

---

## 配置CCR

完成上述安装步骤后，ZCF的完整环境已经成功搭建。接下来的关键一步便是配置CCR以摆脱Anthropic官方高昂资费和地区政策的限制。

当然，在正式配置CCR之前，首要任务是选择合适的大模型服务平台的应用程序接口，以支撑Claude Code运行所产生的高额Token消耗。对于学生用户而言，模型的性价比往往是其优先考量的要素。经过编者的实际测试和综合考量，筛选出以下三个颇具性价比的大模型服务平台，供读者权衡选用：

- **[智谱AI开放平台](https://open.bigmodel.cn)**：该平台推出的[GLM Coding Plan](https://bigmodel.cn/glm-coding)性价比极高。以每月20元的Lite套餐为例，用户可以享受每5小时最多约120次的调用额度，相当于Anthropic官方Pro套餐额度的3倍。其缺点在于模型种类有限，经典的GLM-4.5模型对于Claude Code的工具调用功能支持良好，但是上下文长度仅为128K；而新推出的GLM-4.6模型虽然将上下文长度提升至200K，但是存在容易遗忘和工具调用支持不佳的问题。
- **[硅基流动](https://cloud.siliconflow.cn)**：该聚合平台提供邀请奖励机制。用户可以通过在淘宝、闲鱼等渠道，以大约90元的价格换取大约1400元的赠送额度，随用随付，灵活性较高。其缺点在于使用赠送额度的模型输出速度较慢，在执行以长思考为主的编程任务中，可能会导致效率降低。
- **[云雾API](https://yunwu.ai/register?aff=bxvJ)**：该聚合平台基于中转方案，提供了极高的模型调用速度和相较于官方大幅降低的资费。其缺点在于选择不同的分组渠道时，可能需要配置不同的统一资源定位符（Uniform Resource Locator，URL）。

### 启动CCR服务

打开Windows PowerShell终端，输入以下命令打开CCR管理页面：

```bash
npx zcf ccr
```

<div style="text-align: center;">
  <img src="figures/figure5.png" alt="CCR管理页面" width="70%"/>
  <br />
  CCR管理页面
</div>

在CCR管理页面输入选项“2”并回车打开CCR UI，注意首次打开CCR UI时需要使用登录密钥“sk-zcf-x-ccr”。

<div style="text-align: center;">
  <img src="figures/figure6.png" alt="CCR UI" width="70%"/>
  <br />
  CCR UI
</div>

### 填写供应商

在CCR UI的供应商选项卡中，选择添加供应商，根据所选的大模型服务平台，填写对应信息：

- **以智谱AI开放平台为例，填写流程如下：**
  1. 在“从模板导入”处选择“智谱 Coding Plan”
  2. 在“API密钥”处填写从智谱AI开放平台获取的API密钥
  3. 在“模型”处填写所需调用的模型名称，推荐：“glm-4.6”、“glm-4.5-air”
  4. 点击“保存”

- **以硅基流动为例，填写流程如下：**
  1. 在“从模板导入”处选择“siliconflow”
  2. 在“API密钥”处填写从硅基流动获取的API密钥
  3. 在“模型”处填写所需调用的模型名称，推荐：“moonshotai/Kimi-K2-Thinking”、“MiniMaxAI/MiniMax-M2”、“zai-org/GLM-4.5”、“deepseek-ai/DeepSeek-V3.2-Exp”
  4. 点击“保存”

- **以云雾API为例，填写流程如下：**
  1. 在“名称”处填写“云雾API Claude Code”
  2. 在“API完整地址”处填写“https://yunwu.ai/v1/messages”
  3. 在“API密钥”处填写从云雾API获取的API密钥，并在云雾API配置分组为“Claude Code专属”
  4. 在“模型”处填写所需调用的模型名称，推荐：“claude-sonnet-4-5-20250929”、“claude-sonnet-4-5-20250929-thinking”
  5. 在“供应商转换器”处选择“Anthropic”
  6. 点击“保存”

- **提示**：
  1. 常见的大模型服务平台可以在“从模板导入”中获取对应的模板，之后依次填写“API密钥”、“模型”等信息后点击“保存”即可
  2. 特殊的大模型服务平台需要手动填写所有信息，具体而言：OpenAI兼容的服务平台在“API完整地址”处应以“completions”结尾，Anthropic兼容的服务平台在“API完整地址”处应以“messages”结尾，此外还有一种特殊情况是Codex兼容的服务平台，其在“API完整地址”处应以“responses”结尾

### 填写路由

在CCR UI的路由选项卡中，填写对应信息：

<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
  <thead>
    <tr style="background-color: #f6f8fa; text-align: center; font-weight: bold;">
      <th style="padding: 14px 12px; border: 1px solid #e1e4e8; width: 15%;">路由类型</th>
      <th style="padding: 14px 12px; border: 1px solid #e1e4e8; width: 22%;">适用任务</th>
      <th style="padding: 14px 12px; border: 1px solid #e1e4e8; width: 38%;">模型选择逻辑</th>
      <th style="padding: 14px 12px; border: 1px solid #e1e4e8; width: 25%;">推荐模型</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center; font-weight: bold;">默认</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8;">通用任务</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; line-height: 1.8;">高性能、中上下文、短思考</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center;">GLM-4.5</td>
    </tr>
    <tr style="background-color: #fafbfc;">
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center; font-weight: bold;">后台</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8;">低优先级任务</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; line-height: 1.8;">低成本</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center;">
        DeepSeek-V3.2-Exp
      </td>
    </tr>
    <tr>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center; font-weight: bold;">思考</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8;">高复杂度任务</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; line-height: 1.8;">高性能、长上下文、长思考</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center;">Kimi-K2-Thinking</td>
    </tr>
    <tr style="background-color: #fafbfc;">
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center; font-weight: bold;">长上下文</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8;">长上下文任务</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; line-height: 1.8;">高性能、长上下文</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center;">MiniMax-M2</td>
    </tr>
  </tbody>
</table>

---

至此，恭喜读者你已成功安装ZCF并配置CCR，可以开始使用Claude Code进行高效的程序设计和开发了。接下来的章节将介绍如何使用Claude Code完成实际的编程任务。

---

## 使用Claude Code

本章节基于ZCF的项目说明，概述Claude Code的核心逻辑、关键命令和使用案例，帮助我读者迅速上手Claude Code。

### 核心逻辑

Claude Code 的核心逻辑包含自然语言驱动、工作流协作和上下文管理三大要素。用户通过自然语言触发集成工作流，智能体随即进入协作阶段，依次完成研究、规划、编码、审查等任务；在此过程中，系统调用多种 MCP 服务，增强上下文感知，确保智能体深刻理解任务背景。

### 关键命令

ZCF为Claude Code扩展了命令系统，按任务类型选用，可显著提升开发结构化与效率。关键命令具体如下：

<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
  <thead>
    <tr style="background-color: #f6f8fa; text-align: center; font-weight: bold;">
      <th style="padding: 14px 12px; border: 1px solid #e1e4e8; width: 30%;">命令名称</th>
      <th style="padding: 14px 12px; border: 1px solid #e1e4e8; width: 20%;">命令功能</th>
      <th style="padding: 14px 12px; border: 1px solid #e1e4e8; width: 50%;">具体行为</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center; font-family: 'Courier New', monospace;">/zcf:init-project</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8;">项目初始化</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; line-height: 1.8;">扫描项目目录结构并生成CLAUDE.md等文件</td>
    </tr>
    <tr style="background-color: #fafbfc;">
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center; font-family: 'Courier New', monospace;">/zcf:feat</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8;">功能开发</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; line-height: 1.8;">结构化执行功能开发任务</td>
    </tr>
    <tr>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center; font-family: 'Courier New', monospace;">/compact</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8;">压缩上下文</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; line-height: 1.8;">压缩当前会话上下文</td>
    </tr>
    <tr style="background-color: #fafbfc;">
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center; font-family: 'Courier New', monospace;">/clear</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8;">清除上下文</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; line-height: 1.8;">清空当前会话上下文</td>
    </tr>
    <tr>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center; font-family: 'Courier New', monospace;">/exit</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8;">安全退出</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; line-height: 1.8;">退出当前会话并保留上下文</td>
    </tr>
    <tr>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; text-align: center; font-family: 'Courier New', monospace;">/zcf:git-commit</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8;">版本提交</td>
      <td style="padding: 14px 12px; border: 1px solid #e1e4e8; line-height: 1.8;">自动生成规范commit信息并推送至GitHub</td>
    </tr>

  </tbody>
</table>

### 使用流程

本小节以基于Python实现“即插即用的多种扩散模型实现”的简单项目为例，演示如何使用Claude Code进行开发。

- **准备工作**：
  1. 创建项目文件夹，例如“diffusion_model_implementations”
  2. 确认项目需求和代码风格规范，分别保存为“[idea.md](https://github.com/HIT-SudoMaker/tutorials_for_claude_code/idea.md)”和“[coding_paradigm.md](https://github.com/HIT-SudoMaker/tutorials_for_claude_code/coding_paradigm.md)”

- **启动CCR**：
  1. 打开Windows PowerShell终端，输入`npx zcf ccr`命令打开CCR管理页面
  2. 在CCR管理页面输入选项“4”并回车启动CCR服务

- **启动Python环境**：
  1. 确保自己的Conda终端，例如Anaconda Prompt和Miniforge Prompt，已与命令行终端（cmd）同步，如果未同步可以通过输入`conda init cmd.exe`命令后重启终端实现同步
  2. 打开Conda终端或命令行终端，输入`conda activate env_name`命令激活对应Python环境

- **切换至项目目录**：
  1. 在Conda终端或命令行终端中，输入`cd path\to\diffusion_model_implementations`命令切换至项目目录

- **初始化项目**：
  1. 在Conda终端或命令行终端中，输入`claude`命令启动Claude Code
  2. 在Claude Code交互界面中，输入命令`/zcf:init-project`并回车，Claude Code将自动扫描项目目录结构并生成CLAUDE.md等文件
  3. 在Claude Code交互界面中，输入命令`/zcf:feat 请根据idea.md和coding_paradigm.md的内容，帮助我实现即插即用的多种扩散模型`即可触发Claude Code的开发工作流

- **喜欢喝茶，拒绝抽烟**：
  1. 请勿关闭Claude Code交互界面，等待Claude Code完成任务
  2. 不时检查交互界面的工作流输出状态，学会判断Claude Code是否处于后台输出状态还是已经出现卡死或者结束问题，前者交互界面的工作流阶段会有动态特效，后者交互界面的工作流阶段会静止不动，需要双击`Esc`键中断并重新输入命令
  3. 不时检查交互界面的CCometixLine上下文状态，根据其中的上下文使用情况，适度使用`/compact`命令或者`/clear`命令压缩或者清除上下文
  4. 任务完成后，输入`/exit`命令安全退出Claude Code交互界面

<div style="text-align: center;">
  <img src="figures/figure7.png" alt="Claude Code交互界面" width="70%"/>
  <br />
  Claude Code交互界面
</div>

---