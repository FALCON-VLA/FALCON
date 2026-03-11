<div align="center">

<p>
    <img src="imgs/falcon_logo.png" alt="FALCON_logo" width="40%" height="auto">
</p>

#  | *FALCON* | From Spatial to Actions: <br>Grounding Vision-Language-Action Model in Spatial Foundation Priors (ICLR 2026)

<a href="https://arxiv.org/abs/2510.17439" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-FALCON-red?logo=arxiv" height="25" />
</a>
<a href="https://falcon-vla.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-falcon.io-blue.svg" height="25" />
</a>
<a href="https://huggingface.co/papers/2510.17439" target="_blank">
    <img alt="HF Paper: FALCON" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Paper-FALCON-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/FALCON-VLA" target="_blank">
    <img alt="HF Model: FALCON" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-FALCON-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<!-- <a href="https://huggingface.co/datasets/robovlms/bytedance_robot_benchmark_20" target="_blank">
    <img alt="HF Dataset: BDRBench-20" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Dataset-BDRBench20-ffc107?color=ffc107&logoColor=white" height="25" />
</a> -->
<br>
<a href="https://www.python.org/" target="_blank">
    <img alt="Python 3.8" src="https://img.shields.io/badge/Python-%3E=3.8-blue" height="25" />
</a>
<a href="https://pytorch.org/" target="_blank">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E=2.1-orange" height="25" />
</a>

</div>

<div align="center">
    <br>
<div style="text-align: center;">
    <a href="https://scholar.google.com/citations?user=8nrJ1vsAAAAJ&hl=en"  target="_blank">Zhengshen Zhang</a> &emsp;
    <a href="https://scholar.google.com/citations?user=4dokjDoAAAAJ&hl=zh-CN"  target="_blank">Hao Li</a> &emsp;
    <a href="https://scholar.google.com/citations?user=6XyNVowAAAAJ&hl=en"  target="_blank">Yalun Dai</a> &emsp;
    <a href="https://scholar.google.com/citations?user=ozatRA0AAAAJ&hl=zh-CN"  target="_blank">Zhengbang Zhu</a> &emsp;
    <a href="https://scholar.google.com/citations?user=VhToj4wAAAAJ&hl=zh-CN"  target="_blank">Lei Zhou</a> &emsp;
    <br>
    <a href="https://sg.linkedin.com/in/liu-chenchen"  target="_blank">Chenchen Liu</a> &emsp;
    <a href=""  target="_blank">Dong Wang</a> &emsp;
    <a href="https://scholar.google.com/citations?user=mfH9UFIAAAAJ&hl=en"  target="_blank">Francis E. H. Tay</a> &emsp;
    <a href="https://ch3cook-fdu.github.io/"  target="_blank">Sijin Chen</a> &emsp;
    <br>
    <a href="https://liuziwei7.github.io/"  target="_blank">Ziwei Liu</a> &emsp;
    <a href="https://scholar.google.com/citations?user=i8wNtSgAAAAJ&hl=en"  target="_blank">Yuxiao Liu</a><sup>*</sup><sup>&dagger;</sup> &emsp;
    <a href="https://scholar.google.com/citations?user=laOWyTQAAAAJ&hl=zh-CN"  target="_blank">Xinghang Li</a><sup>*</sup> &emsp;
    <a href="https://panzhous.github.io/"  target="_blank">Pan Zhou</a><sup>*</sup> &emsp;
    <br>
    <p style="text-align: center; margin-bottom: 0;">
        <span class="author-note"><sup>*</sup>Corresponding Author</span>&emsp;
        <span class="author-note"><sup>&dagger;</sup>Project Lead</span>
    </p>
<br>
<p style="text-align: center;">
    ByteDance Seed <br> 
    National University of Singapore &emsp; Nanyang Technological University <br>
    Tsinghua University &emsp; Singapore Management University</p>
</div>
</div>

<hr>

<p>
    <img src="imgs/falcon_teaser.png" alt="FALCON_teaser" width="100%" height="auto">
</p>

## Updates 🚀🚀🚀
- [26/01/2026] 🎊 Thrilled to share that our paper has been accepted to ICLR 2026! Code will be open-sourced soon. Stay tuned!

- [20/10/2025] Existing vision-language-action (VLA) models act in 3D real-world but are typically built on 2D encoders, leaving a spatial reasoning gap that limits generalization and adaptability. In this work, we introduce **FALCON (From Spatial to Action)**, a novel paradigm that injects rich 3D spatial tokens into the action head of a VLA model, enabling robust spatial understanding and SOTA performance across diverse manipulation tasks without disrupting vision-language alignment. See our paper at [here](https://arxiv.org/abs/2510.17439).

## Contents
- [Benchmark Performance Comparison](#benchmark-performance-comparison)
- [TODO List](#todo-list)
- [Citation](#citation)

## 💪 Benchmark Performance Comparison <a name="benchmark-performance-comparison"></a>
### CALVIN Benchmark
![calvin](./imgs/calvin_performance.png "CALVIN Performance")

### SimplerEnv WidowX Robot Experiments
![simpler](./imgs/simpler_bridge.png "SimplerEnv Bridge Performance")

### SimplerEnv Google Robot Experiments
![simpler](./imgs/simpler_gr.png "SimplerEnv Google Robot Performance")

### Real-World Experiments
![real-world](./imgs/real_base_tasks.png "Real-World Performance")
💡 For more sim/real-world benchmark results, please refer to our paper.

## 🗒️ TODO List <a name="todo-list"></a>
- [ ] Release the code, model of FALCON.
- [ ] Release the CALVIN & SimplerEnv evaluation code and model weights for FALCON series.
- [ ] Release pre-training / fine-tuning code for FALCON series.
- [ ] Release the code for real-world deployment of FALCON via [ManiUniCon](https://github.com/Universal-Control/ManiUniCon).

## 🖊️ Citation <a name="citation"></a>
If you find this project useful in your research, please consider cite:
```BibTeX
@article{zhang2025spatial,
  title={From spatial to actions: Grounding vision-language-action model in spatial foundation priors},
  author={Zhang, Zhengshen and Li, Hao and Dai, Yalun and Zhu, Zhengbang and Zhou, Lei and Liu, Chenchen and Wang, Dong and Tay, Francis EH and Chen, Sijin and Liu, Ziwei and others},
  journal={arXiv preprint arXiv:2510.17439},
  year={2025}
}
```
