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

## Referensi
- [DCGAN Paper](https://arxiv.org/abs/1511.06434)