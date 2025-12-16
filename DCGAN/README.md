# Deep Convolutional GAN
Sama seperti GAN tetapi lapisannya diperdalam dengan lapisan konvolusi yang membuatnya cocok untuk data gambar

## Key Concepts
- Memperkenalkan `transposed convolution`
- `Conv2D` : Downsampling, `ConvTranspose2D` : Upsampling (membalik proses konvolusi)
- Dan `batch normalization` untuk stabilitas pelatihan sehingga pada layer bias di set ke False (`bias=False`)
- Tidak ada pooling layer, disini menggunakan `stride` untuk down/up samping
- Tidak ada FC layer

Untuk **stabilitas** ada beberapa kunci di arsitektur DCGAN (sumber: paper asli):
- Pada generator menggunakan fungsi aktivasi ReLU di semua lapisan, kecuali output yang menggunakan TanH
- Pada discriminator semua lapisan menggunakan fungsi aktivasi LeakyReLU

## Notes
### Untuk gambar asli
output_real = D(real_images)  
loss_real = loss_fn(output_real, torch.ones_like(output_real))

### Untuk gambar palsu
output_fake = D(fake_images)  
loss_fake = loss_fn(output_fake, torch.zeros_like(output_fake))

### Total loss
loss_D = (loss_real + loss_fake) / 2


## Referensi
- [DCGAN Paper](https://arxiv.org/abs/1511.06434)