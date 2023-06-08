
### Deblurring Masked Autoencoder is Better Recipe for Ultrasound Image Recognition


This is an official implementation of the paper Deblurring Masked Autoencoder is Better Recipe for Ultrasound Image Recognition

### Abstract
Masked autoencoder (MAE) has attracted unprecedented attention and achieves remarkable performance in many vision tasks. It reconstructs random masked image patches (known as proxy task) during pretraining and learns meaningful semantic representations that can be transferred to downstream tasks. However, MAE has not been thoroughly explored in ultrasound imaging. In this work, we investigate the potential of MAE for ultrasound image recognition. Motivated by the unique property of ultrasound imaging in high noise-to-signal ratio, we propose a novel deblurring MAE approach that incorporates deblurring into the proxy task during pretraining. The addition of deblurring facilitates the pretraining to better recover the subtle details presented in the ultrasound images, thus improving the performance of the downstream classification task. Our experimental results demonstrate the effectiveness of our deblurring MAE, achieving state-of-the-art performance in ultrasound image classification. Overall, our work highlights the potential of MAE for ultrasound image recognition and presents a novel approach that incorporates deblurring to further improve its effectiveness. 

### Method
![tenser](figure/main.drawio.png)

### Pre-trained checkpoints
We use 280,000 thyroid ultrasound images for the pretraining. The following table provides the pre-trained checkpoints:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>8cad7c</tt></td>
<td align="center"><tt>b8b06e</tt></td>
<td align="center"><tt>9bdbb0</tt></td>
</tr>
</tbody></table>

### Fine-tuning for classification

### Fine-tuning for segmentation


### Acknowledgement
The pretraining and finetuning of our project are based on [MAE](https://github.com/facebookresearch/mae) and [ConvMAE](https://github.com/Alpha-VL/ConvMAE). The segmentation part are based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [mae_segmentation](https://github.com/implus/mae_segmentation). Thanks for their wonderful work.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.