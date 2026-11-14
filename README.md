# CourtLine 网页实时人体与球拍识别

在浏览器本地运行，无视频上传。Chrome/Edge优先，支持摄像头和本地视频。
GitHub Pages使用main分支根目录发布本目录，无需Python或付费推理服务器。

本目录含完整浏览器源码与模型：body.onnx、racket.onnx、racket-pose.part*.bin。
球拍关键点大模型按序拼接后加载；切勿删除分块文件。
算法YOLO26人体＋YOLO11球拍检测＋RacketVision RTMPose五关键点。
默认416/640/256输入；低置信度或遮挡时不画球拍轴线；非真实三维测量。
首次下载约85MB资源，后续由浏览器缓存。速度取决于电脑，后台标签页会被浏览器限速。

源码按AGPL-3.0发布，第三方许可证见对应文件。模型来源、转换方法见THIRD_PARTY_NOTICES.md及export_models.py。
模型不会将画面发送到服务器，没有账号或分析日志收集。
