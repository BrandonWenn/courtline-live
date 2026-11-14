# Third-party sources

- Ultralytics YOLO26n-pose and YOLO11n: https://github.com/ultralytics/ultralytics . AGPL-3.0; model exports are derived from the official pretrained checkpoints. Complete application source is included in this repository. See LICENSE and dist/licenses/Ultralytics-AGPL-3.0.txt.
- RacketVision RTMPose-M: https://github.com/OrcustD/RacketVision . MIT project; checkpoint best_PCK_epoch_90.pth from https://huggingface.co/linfeng302/RacketVision-Models . Exported to racket-pose.onnx without retraining; no claim of authorship of the pretrained model. See dist/licenses/RacketVision-MIT.txt.
- MMPose / RTMPose implementation: https://github.com/open-mmlab/mmpose . Apache-2.0. See dist/licenses/MMPose-Apache-2.0.txt.
- ONNX Runtime Web 1.23.2: https://github.com/microsoft/onnxruntime . MIT. Distributed browser runtime is copied from the official npm onnxruntime-web package.

Third-party licenses apply independently to their respective code and model assets. No training data or user videos are redistributed.
